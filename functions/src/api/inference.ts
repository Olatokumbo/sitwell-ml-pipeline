import * as functions from "firebase-functions";
import * as admin from "firebase-admin";
import { PredictionServiceClient, protos } from "@google-cloud/aiplatform";
import { env } from "../config";

const client = new PredictionServiceClient();

/**
 * HTTPS endpoint that performs inference using the user's personalized model.
 */
export const inference = functions.https.onRequest(
  { memory: "4GiB" },
  async (req, res) => {
    try {
      const { gtrace, conf, userId } = req.body as {
        userId: string;
        gtrace: { backrest: number[]; seat: number[] };
        conf: { backrest: number[]; seat: number[] };
      };

      if (!userId || !gtrace || !conf) {
        res.status(400).send({ error: "Missing required fields" });
        return;
      }

      const userDoc = await admin
        .firestore()
        .collection("users")
        .doc(userId)
        .get();

      if (!userDoc.exists) {
        res.status(404).send({ error: "User not found" });
        return;
      }

      const userData = userDoc.data();
      const gtraceEndpointId = userData?.gtraceEndpointId;
      const conformatEndpointId = userData?.conformatEndpointId;

      if (!gtraceEndpointId || !conformatEndpointId) {
        res.status(400).send({ error: "One or both models are not deployed" });
        return;
      }

      // Helper to convert array → protobuf list
      const makeListValue = (
        arr: number[]
      ): protos.google.protobuf.IListValue => ({
        values: arr.map((n) => ({ numberValue: n })),
      });

      // Prepare instances for both models
      const gtraceInstance: protos.google.protobuf.IValue = {
        structValue: {
          fields: {
            backrest: { listValue: makeListValue(gtrace.backrest) },
            seat: { listValue: makeListValue(gtrace.seat) },
          },
        },
      };

      const conformatInstance: protos.google.protobuf.IValue = {
        structValue: {
          fields: {
            backrest: { listValue: makeListValue(conf.backrest) },
            seat: { listValue: makeListValue(conf.seat) },
          },
        },
      };

      const gtraceEndpoint = `projects/${env.GCLOUD_PROJECT_ID}/locations/${env.GCLOUD_REGION}/endpoints/${gtraceEndpointId}`;
      const conformatEndpoint = `projects/${env.GCLOUD_PROJECT_ID}/locations/${env.GCLOUD_REGION}/endpoints/${conformatEndpointId}`;

      // 🔄 Run both predictions in parallel
      const [gtraceResponse, conformatResponse] = await Promise.all([
        client.predict({
          endpoint: gtraceEndpoint,
          instances: [gtraceInstance],
        }),
        client.predict({
          endpoint: conformatEndpoint,
          instances: [conformatInstance],
        }),
      ]);

      res.status(200).send({
        message: "Inference successful",
        gtracePrediction: gtraceResponse[0].predictions,
        conformatPrediction: conformatResponse[0].predictions,
      });
    } catch (error: any) {
      console.error("❌ Inference error:", error);
      res.status(500).send({ error: error.message });
      return;
    }
  }
);
