import { v1 as aiplatform } from "@google-cloud/aiplatform";
import { Storage } from "@google-cloud/storage";
import { getOrThrow } from "../common/helper";

const GCLOUD_REGION = getOrThrow("GCLOUD_REGION", "europe-west2");
const firebaseConfig = process.env.FIREBASE_CONFIG ?
  JSON.parse(process.env.FIREBASE_CONFIG) :
  {};

export const GCLOUD_PROJECT_ID = firebaseConfig.projectId || "";
export const STORAGE_BUCKET = firebaseConfig.storageBucket || "";

const storage = new Storage();
const pipelinesClient = new aiplatform.PipelineServiceClient({
  apiEndpoint: `${GCLOUD_REGION}-aiplatform.googleapis.com`,
});

export { pipelinesClient, storage };

export const env = {
  GCLOUD_REGION,
  GCLOUD_PROJECT_ID,
  STORAGE_BUCKET,
};
