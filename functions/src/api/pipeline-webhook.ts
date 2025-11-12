import * as admin from "firebase-admin";
import * as functions from "firebase-functions";

/**
 * Webhook to receive pipeline completion notifications
 */
export const pipelineWebhook = functions.https.onRequest(
  {
    memory: "1GiB",
  },
  async (req, res) => {
    // Verify request is from Google Cloud (implement proper authentication)

    try {
      const { pipelineJobName, state, userId } = req.body;

      console.log(`📢 Pipeline webhook: ${pipelineJobName} - ${state}`);

      if (state === "PIPELINE_STATE_SUCCEEDED") {
        // Update Firestore
        await admin
          .firestore()
          .collection("calibration_sessions")
          .doc(userId)
          .update({
            status: "training_complete",
            lastUpdated: admin.firestore.FieldValue.serverTimestamp(),
          });

        console.log(`✅ Training completed for user ${userId}`);
      } else if (state === "PIPELINE_STATE_FAILED") {
        await admin
          .firestore()
          .collection("calibration_sessions")
          .doc(userId)
          .update({
            status: "training_failed",
            lastUpdated: admin.firestore.FieldValue.serverTimestamp(),
          });

        console.log(`❌ Training failed for user ${userId}`);
      }

      res.status(200).send("OK");
    } catch (error) {
      console.error("❌ Error processing pipeline webhook:", error);
      res.status(500).send("Error processing webhook");
    }
  }
);
