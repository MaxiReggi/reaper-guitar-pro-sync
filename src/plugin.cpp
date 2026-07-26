#include "plugin.h"

#include "guitar_pro.h"
#include "reaper.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>

namespace tnt {

// REAPER runs this periodically 30 times/second.
static constexpr double MINIMUM_TIME_STEP = 0.001;              // Seconds
static constexpr double MINIMUM_PLAY_RATE_STEP = 0.001;         // Seconds
static constexpr double GUITAR_PRO_CURSOR_JUMP_THRESHOLD = 0.1; // Seconds
static constexpr double LOOP_BOUNDARY_TOLERANCE = 0.06;         // Seconds; don't touch anything this close to a loop edge
// Cap dead reckoning extrapolation to avoid runaway drift between Guitar Pro updates.
static constexpr double DEAD_RECKONING_MAX_EXTRAPOLATION = 0.2; // Seconds

// Position servo: instead of waiting for drift to accumulate and then jumping REAPER's
// cursor (which leaves a window where the notation is visibly out of sync with the audio
// before the jump happens), continuously nudge REAPER's play rate a tiny, inaudible amount
// toward Guitar Pro's position every tick. This keeps the drift from ever growing large
// enough to be noticeable, instead of correcting it after the fact.
static constexpr double POSITION_SERVO_MAX_RATE_NUDGE = 0.004;     // max +-0.4% deviation from nominal tempo
static constexpr double POSITION_SERVO_JUMP_THRESHOLD  = 0.25;     // Seconds; beyond this, drift is treated as a
                                                                    // real discontinuity (stall/glitch) - jump instead
// Error at which the nudge saturates at POSITION_SERVO_MAX_RATE_NUDGE. Keep this well below
// POSITION_SERVO_JUMP_THRESHOLD so typical small drift gets a strong, fast correction instead
// of a barely-there one, without hammering REAPER's preserve-pitch time stretch by re-issuing
// the rate too often/aggressively (0.03s was tried and caused audible artifacts).
static constexpr double POSITION_SERVO_SATURATION_ERROR = 0.09;   // Seconds
static constexpr double POSITION_SERVO_GAIN = POSITION_SERVO_MAX_RATE_NUDGE / POSITION_SERVO_SATURATION_ERROR;
// Only re-issue SetPlayRate when the target moved by at least this much, so the time
// stretch engine isn't asked to recompute on every single tick for negligible changes.
static constexpr double POSITION_SERVO_UPDATE_STEP = 0.0015;

// Small hardcoded bias to compensate a residual lag that may show up consistently, rather
// than varying take to take like normal drift — a fixed lead is the right tool for that
// (the servo/dead-reckoning only correct relative drift, not a constant offset). This value
// was tuned by ear for GoPlayAlong on a specific machine/audio setup — re-tune it here from
// scratch (start at 0.0) rather than trusting the ported number.
static constexpr double EXTRA_LEAD_COMPENSATION = 0.0;   // Seconds

struct Plugin::Impl final {
    Impl(PluginState& plugin_state)
        : m_plugin_state(plugin_state)
    {}

    void MainLoop()
    {
        try
        {
            // Read current Guitar Pro and REAPER states
            m_guitar_pro_state = m_guitar_pro.ReadProcessMemory();
        }
        catch (const std::runtime_error& error)
        {
            if (m_last_error != error.what())
            {
                m_reaper.ShowConsoleMessage(error.what());
                m_last_error = error.what();
            }

            return;
        }

        if (!m_last_error.empty())
        {
            m_reaper.ShowConsoleMessage("Successfully connected to Guitar Pro process.\n");
            m_last_error = "";
        }

        this->UpdateDeadReckoning();

        // Ensure REAPER stays in sync while Guitar Pro is playing
        if (m_guitar_pro_state.play_state)
        {
            this->SyncLoopState();
            this->SyncTimeSelection();
            this->SyncPlayPosition();
            this->SyncPlayRate();
        }

        // Allow some control while Guitar Pro and REAPER are both paused
        else if (this->ReaperStoppedOrPaused())
        {
            // Sync loop state
            if (this->GuitarProLoopStateChanged())
            {
                this->SyncLoopState();
            }

            // Sync time selection and cursor
            if (this->GuitarProTimeSelectionChanged() && m_guitar_pro_state.time_selection_end_position > MINIMUM_PLAY_RATE_STEP)
            {
                this->SyncTimeSelection();
                this->SetPlayPosition(m_guitar_pro_state.time_selection_start_position);
            }
            else if (this->GuitarProCursorMoved())
            {
                this->SyncTimeSelection();
                this->SetPlayPosition(m_guitar_pro_state.play_position);
            }

            // Sync play rate. SyncPlayRate() gates internally on GuitarProPlayRateChanged()
            // (see its definition), so this can be called unconditionally.
            // TODO this doesn't work while paused because the value read from memory only updates at runtime.
            // We need to find a new memory address to get this to work more effectively
            this->SyncPlayRate();
        }

        // Ensure REAPER is playing if Guitar Pro is playing
        this->SyncPlayState();

        // Save previous Guitar Pro state
        m_prev_guitar_pro_state = m_guitar_pro_state;
    }

private:
    // Dead reckoning: track the last Guitar Pro position update and extrapolate forward
    // using elapsed real time x play rate. This gives a smooth real-time estimate of
    // Guitar Pro's current position between its ~15 Hz memory updates, so the position
    // servo has a stable error signal to correct against instead of stale/jumpy data.
    void UpdateDeadReckoning()
    {
        if (!this->CompareDoubles(m_guitar_pro_state.play_position, m_gp_reckoned_position, MINIMUM_TIME_STEP))
        {
            m_gp_reckoned_position = m_guitar_pro_state.play_position;
            m_gp_reckoned_time = std::chrono::steady_clock::now();
            m_gp_reckoning_valid = true;
        }
    }

    double GetDeadReckonedPosition() const
    {
        if (!m_gp_reckoning_valid || !m_guitar_pro_state.play_state)
            return m_guitar_pro_state.play_position;

        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - m_gp_reckoned_time).count();
        const double capped = elapsed < DEAD_RECKONING_MAX_EXTRAPOLATION ? elapsed : DEAD_RECKONING_MAX_EXTRAPOLATION;
        return m_gp_reckoned_position + capped * m_guitar_pro_state.play_rate;
    }

    void SyncLoopState()
    {
        // Sync the loop state (unless we are playing and there is a count in timer)
        if (m_guitar_pro_state.loop_state && !(m_guitar_pro_state.play_state && m_guitar_pro_state.count_in_state))
        {
            m_reaper.SetRepeat(true);
        }
        else
        {
            m_reaper.SetRepeat(false);
        }
    }

    void SyncTimeSelection()
    {
        // Sync the time selection
        m_reaper.SetTimeSelection(m_guitar_pro_state.time_selection_start_position, m_guitar_pro_state.time_selection_end_position);
    }

    void SyncPlayPosition()
    {
        const double gp_pos = this->GetDeadReckonedPosition();
        const double reaper_pos = m_reaper.GetPlayPosition();

        // Do not touch anything right at a loop boundary — let Guitar Pro's own loop
        // reset settle first, and drop any lingering rate nudge so it doesn't carry over.
        if (this->CompareDoubles(reaper_pos, m_guitar_pro_state.time_selection_start_position, LOOP_BOUNDARY_TOLERANCE)
         || this->CompareDoubles(reaper_pos, m_guitar_pro_state.time_selection_end_position, LOOP_BOUNDARY_TOLERANCE))
        {
            this->ResetPlayRateToNominal();
            return;
        }

        // Follow intentional seeks in Guitar Pro immediately — a servo nudge would take
        // too long to catch up to a deliberate jump, so just cut over.
        if (!this->CompareDoubles(m_prev_guitar_pro_state.play_position, m_guitar_pro_state.play_position, GUITAR_PRO_CURSOR_JUMP_THRESHOLD))
        {
            this->SetPlayPosition(gp_pos + m_reaper.GetOutputLatency() + EXTRA_LEAD_COMPENSATION);
            return;
        }

        const double error = (gp_pos + m_reaper.GetOutputLatency() + EXTRA_LEAD_COMPENSATION) - reaper_pos;

        if (fabs(error) > POSITION_SERVO_JUMP_THRESHOLD)
        {
            // Drift got too large for a smooth correction (e.g. a stall or a glitched
            // read) — fall back to a hard jump rather than nudging for a long time.
            this->SetPlayPosition(gp_pos + m_reaper.GetOutputLatency() + EXTRA_LEAD_COMPENSATION);
            return;
        }

        this->EnablePreservePitch();

        const double correction = std::clamp(error * POSITION_SERVO_GAIN, -POSITION_SERVO_MAX_RATE_NUDGE, POSITION_SERVO_MAX_RATE_NUDGE);
        const double target_rate = m_guitar_pro_state.play_rate * (1.0 + correction);

        if (!this->CompareDoubles(m_reaper.GetPlayRate(), target_rate, POSITION_SERVO_UPDATE_STEP))
        {
            m_reaper.SetPlayRate(target_rate);
        }
    }

    // Cancels any active servo rate nudge, returning REAPER to Guitar Pro's exact nominal tempo.
    void ResetPlayRateToNominal()
    {
        if (m_guitar_pro_state.play_rate > MINIMUM_PLAY_RATE_STEP
         && !this->CompareDoubles(m_reaper.GetPlayRate(), m_guitar_pro_state.play_rate, MINIMUM_PLAY_RATE_STEP))
        {
            m_reaper.SetPlayRate(m_guitar_pro_state.play_rate);
        }
    }

    // Handles deliberate tempo changes in Guitar Pro. Gated on GuitarProPlayRateChanged()
    // rather than comparing against REAPER's live rate, since the position servo
    // intentionally keeps REAPER's live rate slightly off the nominal tempo — comparing
    // directly would fight the servo and pause playback every tick.
    //
    // TODO: The running playback rate memory location seems to take a bit to update when playing the song
    // Because of this, the playback rate may register as 0 for a fraction of a second.
    // Look for a better address in Cheat Engine so this can be done faster
    void SyncPlayRate()
    {
        if (m_guitar_pro_state.play_rate > MINIMUM_PLAY_RATE_STEP && this->GuitarProPlayRateChanged())
        {
            // Always ensure preserve pitch is set before stretching
            this->EnablePreservePitch();

            // REAPER handles stretching much more efficiently if the song is paused
            m_reaper.SetPlayState(ReaperPlayState::PAUSED);
            m_reaper.SetPlayRate(m_guitar_pro_state.play_rate);
        }
    }

    void SyncPlayState()
    {
        if (m_guitar_pro_state.play_state)
        {
            // Stop REAPER if Guitar Pro is currently counting in and the cursor is not moving
            if (m_guitar_pro_state.count_in_state
             && (!this->GuitarProCursorMoved() || (m_guitar_pro_state.time_selection_start_position > MINIMUM_TIME_STEP && m_prev_guitar_pro_state.play_position < MINIMUM_TIME_STEP)))
            {
                // DO NOT cut a loop short
                if (!this->CompareDoubles(m_reaper.GetPlayPosition(), m_guitar_pro_state.time_selection_start_position, MINIMUM_TIME_STEP)
                 && m_reaper.GetPlayPosition() < m_guitar_pro_state.time_selection_end_position)
                {
                    return;
                }

                m_reaper.SetPlayState(ReaperPlayState::STOPPED);
            }

            else if (this->ReaperStoppedOrPaused())
            {
                // If a loop is specified start there
                if (m_guitar_pro_state.time_selection_start_position > MINIMUM_TIME_STEP)
                {
                    this->SetPlayPosition(m_guitar_pro_state.time_selection_start_position + m_reaper.GetOutputLatency() + EXTRA_LEAD_COMPENSATION);
                }

                else
                {
                    this->SetPlayPosition(m_guitar_pro_state.play_position + m_reaper.GetOutputLatency() + EXTRA_LEAD_COMPENSATION);
                }

                m_reaper.SetPlayState(ReaperPlayState::PLAYING);
            }
        }

        // Stop REAPER if Guitar Pro is not playing
        else if (!this->ReaperStoppedOrPaused() && m_prev_guitar_pro_state.play_state)
        {
            // DO NOT cut a time selection short
            if (m_reaper.GetPlayPosition() < m_guitar_pro_state.time_selection_end_position
             && this->CompareDoubles(m_reaper.GetPlayPosition(), m_guitar_pro_state.time_selection_end_position, LOOP_BOUNDARY_TOLERANCE)
             && !this->CompareDoubles(m_reaper.GetPlayPosition(), m_guitar_pro_state.time_selection_start_position, LOOP_BOUNDARY_TOLERANCE))
            {
                m_guitar_pro_state.play_state = true;
                return;
            }

            m_reaper.SetPlayState(ReaperPlayState::STOPPED);
        }
    }

    void SetPlayPosition(const double time)
    {
        m_reaper.SetEditCursorPosition(time, false, true);
    }

    // Returns true if the two values are within epsilon of each other
    bool CompareDoubles(const double val1, const double val2, const double epsilon) const
    {
        return (fabs(val1 - val2) < epsilon);
    }

    bool GuitarProLoopStateChanged() const
    {
        return m_guitar_pro_state.loop_state != m_prev_guitar_pro_state.loop_state;
    }

    bool GuitarProTimeSelectionChanged() const
    {
        return !this->CompareDoubles(m_guitar_pro_state.time_selection_start_position, m_prev_guitar_pro_state.time_selection_start_position, MINIMUM_TIME_STEP)
            || !this->CompareDoubles(m_guitar_pro_state.time_selection_end_position, m_prev_guitar_pro_state.time_selection_end_position, MINIMUM_TIME_STEP);
    }

    bool GuitarProCursorMoved() const
    {
        return !this->CompareDoubles(m_guitar_pro_state.play_position, m_prev_guitar_pro_state.play_position, MINIMUM_TIME_STEP);
    }

    bool GuitarProPlayRateChanged() const
    {
        return !this->CompareDoubles(m_guitar_pro_state.play_rate, m_prev_guitar_pro_state.play_rate, MINIMUM_PLAY_RATE_STEP);
    }

    bool ReaperStoppedOrPaused() const
    {
        switch (m_reaper.GetPlayState())
        {
        case ReaperPlayState::STOPPED:
        case ReaperPlayState::PAUSED:
            return true;
        case ReaperPlayState::PLAYING:
            return false;
        default:
            // This should never happen
            throw std::runtime_error("REAPER is in an invalid play state!\n");
        }
    }

    void EnablePreservePitch() const
    {
        // If Preserve Pitch is OFF, enable it
        if (!m_reaper.GetToggleCommandState(ReaperToggleCommand::PRESERVE_PITCH)) {
            m_reaper.ToggleCommand(ReaperToggleCommand::PRESERVE_PITCH);
        }
    }

    PluginState& m_plugin_state;
    GuitarPro m_guitar_pro;
    Reaper m_reaper;

    GuitarProState m_prev_guitar_pro_state;
    GuitarProState m_guitar_pro_state;

    // Keeps track of the last error (prevents spamming the log with errors)
    std::string m_last_error = "";

    double m_gp_reckoned_position = 0.0;
    std::chrono::steady_clock::time_point m_gp_reckoned_time;
    bool m_gp_reckoning_valid = false;
};

Plugin::Plugin(PluginState& plugin_state)
    : m_impl(std::make_unique<Impl>(plugin_state))
{}

Plugin::~Plugin() = default;

void Plugin::MainLoop()
{
    m_impl->MainLoop();
}

}
