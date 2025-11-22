import { PipelineServiceClient } from "@google-cloud/aiplatform";
import { Storage } from "@google-cloud/storage";
import { getOrThrow } from "../common/helper";

const GCLOUD_REGION = getOrThrow("GCLOUD_REGION", "europe-west2");
const GCLOUD_PROJECT_NUMBER = getOrThrow(
  "GCLOUD_PROJECT_NUMBER",
  "762245799567"
);
const firebaseConfig: {
  databaseURL: string;
  storageBucket: string;
  projectId: string;
} = process.env.FIREBASE_CONFIG ? JSON.parse(process.env.FIREBASE_CONFIG) : {};

export const GCLOUD_PROJECT_ID = firebaseConfig.projectId || "";
export const STORAGE_BUCKET = firebaseConfig.storageBucket || "";

const storage = new Storage();
const pipelinesClient = new PipelineServiceClient({
  apiEndpoint: `${GCLOUD_REGION}-aiplatform.googleapis.com`,
});

export { pipelinesClient, storage };

export const env = {
  GCLOUD_REGION,
  GCLOUD_PROJECT_ID,
  GCLOUD_PROJECT_NUMBER,
  STORAGE_BUCKET,
};
