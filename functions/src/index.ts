import { setGlobalOptions } from "firebase-functions";
import { triggerTraining } from "./api/trigger-training";
import { env } from "./config";
import { dataUploadTrigger } from "./api/data-upload-trigger";
import { pipelineWebhook } from "./api/pipeline-webhook";
import { inference } from "./api/inference";

setGlobalOptions({
  maxInstances: 10,
  region: env.GCLOUD_REGION,
  memory: "1GiB",
});

export { triggerTraining, dataUploadTrigger, pipelineWebhook, inference };
