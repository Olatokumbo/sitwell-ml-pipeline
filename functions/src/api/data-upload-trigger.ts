import { updateCalibrationProgress } from "../services/calibration.service";
import { onObjectFinalized } from "firebase-functions/storage";

/**
 * Triggered when files are uploaded to Firebase Storage
 * Monitors posture data uploads and starts training when complete
 */
export const dataUploadTrigger = onObjectFinalized(async (object) => {
  const filePath = object.data.name;
  const bucketName = object.bucket;

  console.log(`📁 File uploaded: ${filePath}`);
  console.log(`📦 Bucket: ${bucketName}`);

  // Check if this is a user posture file
  if (
    !filePath ||
    !filePath.includes("/posture_data/") ||
    !filePath.endsWith(".npz")
  ) {
    console.log(`⏭️ Ignoring non-posture file: ${filePath}`);
    return null;
  }

  // Extract user ID from path: users/user_123/posture_data/upright.parquet
  const pathParts = filePath.split("/");
  if (pathParts.length < 4 || pathParts[0] !== "users") {
    console.log(`⚠️ Invalid file path format: ${filePath}`);
    return null;
  }

  const userId = pathParts[1];
  const postureName = pathParts[3].replace(".npz", "");

  console.log(`👤 New calibration data for user: ${userId}`);
  console.log(`📄 Posture: ${postureName}`);

  try {
    await updateCalibrationProgress(userId, postureName);
    return null;
  } catch (error) {
    console.error(`❌ Error processing calibration for user ${userId}:`, error);
    throw error;
  }
});
