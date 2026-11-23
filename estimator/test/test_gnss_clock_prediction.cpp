#include <gtest/gtest.h>

#include "utility/gnss_clock_prediction.h"

TEST(GnssClockPredictionTest, UsesElapsedSecondsInsteadOfEpochIndex)
{
    const double start = 1606691920.0;
    EXPECT_DOUBLE_EQ(1000.0, predictGnssClockBias(1000.0, 80.0, start, start));
    EXPECT_NEAR(1008.0, predictGnssClockBias(1000.0, 80.0, start, start + 0.1), 1e-5);
    EXPECT_NEAR(1016.0, predictGnssClockBias(1000.0, 80.0, start, start + 0.2), 1e-5);
    EXPECT_NEAR(992.0, predictGnssClockBias(1000.0, 80.0, start, start - 0.1), 1e-5);
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
