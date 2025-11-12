import { protos } from "@google-cloud/aiplatform";
import * as admin from "firebase-admin";
import { env, pipelinesClient, storage } from "../config";

/**
 * Start the personalized training pipeline for the user
 */
export async function startUserTrainingPipeline(
  userId: string,
  bucketName: string
): Promise<string> {
  const projectId = env.GCLOUD_PROJECT_ID;
  const region = env.GCLOUD_REGION;

  try {
    console.log(`🚀 Starting training pipeline for user: ${userId}`);

    // Create unique pipeline run name
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    const displayName = `user-${userId}-training-${timestamp}`;

    // Pipeline template path
    const templateUri = `https://${region}-kfp.pkg.dev/sitwell/sitwell/posture-pipeline-v1/latest`;

    const parameterValues: { [key: string]: protos.google.protobuf.IValue } = {
      bucket_name: { stringValue: bucketName },
      user_id: { stringValue: userId },
      augmentation_samples: { numberValue: 5 },
      cnn_epochs: { numberValue: 30 },
      cnn_batch_size: { numberValue: 8 },
    };
    // Create pipeline job request
    const pipelineJob: protos.google.cloud.aiplatform.v1.IPipelineJob = {
      displayName,
      templateUri,
      runtimeConfig: {
        gcsOutputDirectory: `gs://${bucketName}/users/${userId}/pipeline_runs`,
        parameterValues,
      },
    };

    // Submit pipeline job
    const parent = `projects/${projectId}/locations/${region}`;
    const [operation] = await pipelinesClient.createPipelineJob({
      parent,
      pipelineJob,
    });

    const jobName = operation.name || displayName;

    console.log(`✅ Pipeline started for user ${userId}`);
    console.log(`📊 Job name: ${displayName}`);
    console.log(
      `🔗 Pipeline root: gs://${bucketName}/users/${userId}/pipeline_runs`
    );

    // Update Firestore
    await admin
      .firestore()
      .collection("calibration_sessions")
      .doc(userId)
      .update({
        status: "training_started",
        lastUpdated: admin.firestore.FieldValue.serverTimestamp(),
      });

    return jobName;
  } catch (error) {
    console.error(`❌ Error starting pipeline for user ${userId}:`, error);

    await admin
      .firestore()
      .collection("calibration_sessions")
      .doc(userId)
      .update({
        status: "training_failed",
        lastUpdated: admin.firestore.FieldValue.serverTimestamp(),
      });

    throw error;
  }
}

/**
 * Check if user has a trained model
 */
export async function checkUserModelExists(userId: string): Promise<boolean> {
  try {
    const bucketName = env.STORAGE_BUCKET;
    const bucket = storage.bucket(bucketName);
    const modelPrefix = `users/${userId}/models/`;

    const [files] = await bucket.getFiles({
      prefix: modelPrefix,
      maxResults: 1,
    });

    return files.length > 0;
  } catch (error) {
    console.error(`❌ Error checking model for user ${userId}:`, error);
    return false;
  }
}
