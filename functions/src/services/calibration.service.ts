import * as admin from "firebase-admin";
import { storage } from "../config";
import { CalibrationSession } from "../common/interface";
import { postures } from "../common/constant";

/**
 * Check if user has completed all required calibration postures
 */
export async function isCalibrationComplete(
  userId: string,
  bucketName: string
): Promise<boolean> {
  try {
    const bucket = storage.bucket(bucketName);
    const userFolder = `users/${userId}/posture_data/`;

    const [files] = await bucket.getFiles({ prefix: userFolder });

    const existingFiles = files
      .filter((file) => file.name.endsWith(".npz"))
      .map((file) => file.name.split("/").pop() || "");

    console.log(`📁 Existing files for ${userId}:`, existingFiles);

    const missingRequired = postures.filter((f) => !existingFiles.includes(f));

    if (missingRequired.length > 0) {
      console.log(`📋 Missing required files for ${userId}:`, missingRequired);
      return false;
    }

    console.log(`🎉 All required calibration files present for ${userId}!`);

    await admin
      .firestore()
      .collection("calibration_sessions")
      .doc(userId)
      .update({
        status: "complete",
        lastUpdated: admin.firestore.FieldValue.serverTimestamp(),
      });

    return true;
  } catch (error) {
    console.error(`❌ Error checking calibration status for ${userId}:`, error);
    return false;
  }
}

/**
 * Get remaining postures needed for calibration
 */
export async function getRemainingPostures(userId: string): Promise<string[]> {
  try {
    const calibrationRef = admin
      .firestore()
      .collection("calibration_sessions")
      .doc(userId);
    const doc = await calibrationRef.get();

    if (doc.exists) {
      const data = doc.data() as CalibrationSession;
      return data.requiredPostures.filter(
        (p) => !data.completedPostures.includes(p)
      );
    }

    return postures;
  } catch (error) {
    console.error(`❌ Error getting remaining postures for ${userId}:`, error);
    return [];
  }
}

/**
 * Update calibration progress in Firestore
 */
export async function updateCalibrationProgress(
  userId: string,
  completedPosture: string
): Promise<void> {
  const calibrationRef = admin
    .firestore()
    .collection("calibration_sessions")
    .doc(userId);

  const doc = await calibrationRef.get();

  if (doc.exists) {
    const data = doc.data() as CalibrationSession;

    // Add completed posture if not already present
    if (!data.completedPostures.includes(completedPosture)) {
      await calibrationRef.update({
        completedPostures:
          admin.firestore.FieldValue.arrayUnion(completedPosture),
        lastUpdated: admin.firestore.FieldValue.serverTimestamp(),
      });
    }
  } else {
    // Create new calibration session
    const newSession: CalibrationSession = {
      userId,
      sessionStart: new Date().toISOString(),
      requiredPostures: postures,
      optionalPostures: [],
      completedPostures: [completedPosture],
      status: "in_progress",
      lastUpdated: new Date().toISOString(),
    };

    await calibrationRef.set(newSession);
  }
}
