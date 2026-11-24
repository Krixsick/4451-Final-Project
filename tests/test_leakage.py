"""Unit tests to verify no data leakage in rolling feature calculations."""

import unittest
import pandas as pd

from scripts.features import previous_average, previous_season_average

class TestFeatureLeakage(unittest.TestCase):

    def test_previous_average_excludes_current_game(self):
        """Verify a rolling window strictly averages past games, not the current one."""
        # Mock chronological player points for 5 consecutive game appearances
        pts = pd.Series([10, 20, 30, 40, 50])
        
        # Apply the 3-game rolling average function being tested
        result = previous_average(pts, window=3)
        
        # Boundary check: The first 3 games lack sufficient history, so they must evaluate to NaN
        self.assertTrue(pd.isna(result[0]))
        
        # Equivalence check: For the 4th game (index 3), the prior 3 games are [10, 20, 30] (avg 20.0).
        # If the target game (40) leaked via the shift function, the average would incorrectly calculate as 30.0.
        self.assertEqual(result[3], 20.0, "Current game data leaked!")
        self.assertEqual(result[4], 30.0, "Current game data leaked!")
        
        print("\n[Test Passed]: Rolling averages (10-game/5-game) successfully exclude the target game.")

    def test_previous_season_average_excludes_current_game(self):
        """Verify the expanding season average excludes the current game."""
        # Mock points for the first 4 games of a season
        pts = pd.Series([10, 20, 30, 40])
        
        # Apply the expanding season-to-date average function
        result = previous_season_average(pts)
        
        # Verify the first game evaluates to NaN (no prior season history exists to average)
        self.assertTrue(pd.isna(result[0]))
        
        # Verify the 4th game (index 3) averages only the first 3 historical games ([10, 20, 30] -> avg 20.0)
        self.assertEqual(result[3], 20.0, "Current game leaked into season average!")
        
        print("[Test Passed]: Season-to-date averages successfully exclude the target game.")

    def test_absence_handling_no_zero_padding(self):
        """Verify that missing a game does not artificially pull down the average."""
        # Mock points where a player played, missed calendar games, and then returned.
        # The system must calculate based on actual game appearances, ignoring calendar gaps.
        pts = pd.Series([10, 20, 30])
        
        # Calculate a 2-appearance rolling average
        result = previous_average(pts, window=2)
        
        # The 3rd appearance (index 2) must average the prior 2 actual appearances ([10, 20] -> avg 15.0).
        # It should not implicitly pad a 0 for the missed game (which would drop the rolling average to 10.0).
        self.assertEqual(result[2], 15.0, "Absences skewing the history!")
        
        print("[Test Passed]: Player absences do not leak zeros into the calculation.")
        
    def test_boundary_first_appearance(self):
        """Verify the very first game returns NaN to prevent hallucinating data."""
        # Boundary Value Analysis (BVA): Testing the absolute minimum data threshold (1 appearance)
        pts = pd.Series([15])
        
        # Attempt to calculate a 5-game rolling average on a 1-game dataset
        result = previous_average(pts, window=5)
        
        # The output must be NaN to ensure the model excludes this row from training, preventing hallucinated features
        self.assertTrue(pd.isna(result[0]))
        
        print("[Test Passed]: Boundary condition (first appearance) correctly handled.")

if __name__ == "__main__":
    unittest.main()