import * as functions from "firebase-functions";
import { env } from "../config/index";
import {
  getRemainingPostures,
  isCalibrationComplete,
} from "../services/calibration.service";
import { startUserTrainingPipeline } from "../services/pipeline.service";

/**
 * HTTP function to manually trigger training for a user
 */
export const triggerTraining = functions.https.onCall(
  {
    memory: "1GiB",
  },
  async (request) => {
    const data = request.data;

    // Verify authentication
    // if (!request.auth) {
    //   throw new functions.https.HttpsError(
    //     "unauthenticated",
    //     "User must be authenticated"
    //   );
    // }

    const { userId } = data;

    if (!userId) {
      throw new functions.https.HttpsError(
        "invalid-argument",
        "Missing userId parameter"
      );
    }

    console.log(`🔧 Manual trigger request for user: ${userId}`);

    try {
      // Check if calibration is complete
      const isComplete = await isCalibrationComplete(
        userId,
        env.STORAGE_BUCKET
      );

      if (isComplete) {
        const jobName = await startUserTrainingPipeline(
          userId,
          env.STORAGE_BUCKET
        );

        return {
          status: "success",
          message: `Training started for user ${userId}`,
          jobName,
          userId,
        };
      } else {
        const remaining = await getRemainingPostures(userId);
        throw new functions.https.HttpsError(
          "failed-precondition",
          `Incomplete calibration data for user ${userId}. 
        Missing: ${remaining.join(", ")}`
        );
      }
    } catch (error) {
      console.error(`❌ Error in manual trigger for user ${userId}:`, error);
      throw new functions.https.HttpsError(
        "internal",
        `Training trigger failed: ${(error as Error).message}`
      );
    }
  }
);
