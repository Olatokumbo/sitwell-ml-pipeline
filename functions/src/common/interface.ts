import { UserRole } from "./enum";

export interface CalibrationSession {
  userId: string;
  sessionStart: string;
  requiredPostures: string[];
  optionalPostures: string[];
  completedPostures: string[];
  status:
    | "in_progress"
    | "complete"
    | "training_started"
    | "training_complete"
    | "training_failed";
  lastUpdated: string;
}

export interface User {
  firstName: string;
  lastName: string;
  role: UserRole;
  age?: number;
  gtraceEndpointId?: string;
  conformatEndpointId?: string;
}

export interface Posture {
  id: string;
  name: string;
  description: string;
  descriptor: string[];
}
