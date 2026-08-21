import { pgTable, serial, text, boolean, integer, timestamp, doublePrecision } from "drizzle-orm/pg-core";

export const usersTable = pgTable("users", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  email: text("email").notNull().unique(),
  passwordHash: text("password_hash").notNull(),
  role: text("role").notNull(), // 'admin' | 'guard' | 'manager'
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type User = typeof usersTable.$inferSelect;
export type InsertUser = typeof usersTable.$inferInsert;

export const vehiclesTable = pgTable("vehicles", {
  id: serial("id").primaryKey(),
  plateNumber: text("plate_number").notNull().unique(),
  vehicleType: text("vehicle_type").notNull(), // 'car' | 'truck' | 'bus' | 'two_wheeler'
  ownerName: text("owner_name").notNull(),
  transporter: text("transporter").notNull(),
  isAuthorized: boolean("is_authorized").default(true).notNull(),
  status: text("status").default("Available").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type VehicleModel = typeof vehiclesTable.$inferSelect;
export type InsertVehicle = typeof vehiclesTable.$inferInsert;

export const driversTable = pgTable("drivers", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  licenseNumber: text("license_number").notNull(),
  phone: text("phone").notNull(),
  vehiclePlate: text("vehicle_plate"),
  status: text("status").default("Available").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type DriverModel = typeof driversTable.$inferSelect;
export type InsertDriver = typeof driversTable.$inferInsert;

export const camerasTable = pgTable("cameras", {
  id: serial("id").primaryKey(),
  name: text("name").notNull(),
  gate: text("gate").notNull(),
  direction: text("direction").notNull(), // 'entry' | 'exit' | 'both'
  status: text("status").default("online").notNull(),
  lastSeen: text("last_seen").default("just now").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type CameraModel = typeof camerasTable.$inferSelect;
export type InsertCamera = typeof camerasTable.$inferInsert;

export const scheduledTripsTable = pgTable("scheduled_trips", {
  id: serial("id").primaryKey(),
  plate: text("plate").notNull(),
  driver: text("driver").notNull(),
  transporter: text("transporter").notNull(),
  gate: text("gate").notNull(),
  purpose: text("purpose").notNull(),
  expectedArrival: text("expected_arrival").notNull(),
  expectedDeparture: text("expected_departure").notNull(),
  status: text("status").notNull(), // 'scheduled' | 'arrived' | 'entry_approved' | 'inside_plant' | 'at_destination' | 'exit_detected' | 'completed' | 'exception'
  entryTime: text("entry_time"),
  dwellMinutes: integer("dwell_minutes"),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type ScheduledTripModel = typeof scheduledTripsTable.$inferSelect;
export type InsertTrip = typeof scheduledTripsTable.$inferInsert;

export const gateEventsTable = pgTable("gate_events", {
  id: serial("id").primaryKey(),
  plate: text("plate").notNull(),
  eventType: text("event_type").notNull(), // 'entry' | 'exit'
  gate: text("gate").notNull(),
  camera: text("camera").notNull(),
  decision: text("decision").notNull(), // 'allow' | 'deny' | 'manual_review'
  confidence: doublePrecision("confidence").notNull(),
  timestamp: text("timestamp").notNull(),
  vehicleType: text("vehicle_type").notNull(),
  transporter: text("transporter").notNull(),
  isCorrected: boolean("is_corrected").default(false).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type GateEventModel = typeof gateEventsTable.$inferSelect;
export type InsertGateEvent = typeof gateEventsTable.$inferInsert;

export const alertsTable = pgTable("alerts", {
  id: serial("id").primaryKey(),
  type: text("type").notNull(),
  severity: text("severity").notNull(), // 'high' | 'medium' | 'low'
  message: text("message").notNull(),
  plate: text("plate").notNull(),
  gate: text("gate").notNull(),
  time: text("time").notNull(),
  isRead: boolean("is_read").default(false).notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type AlertModel = typeof alertsTable.$inferSelect;
export type InsertAlert = typeof alertsTable.$inferInsert;

export const reviewQueueTable = pgTable("review_queue", {
  id: serial("id").primaryKey(),
  plate: text("plate").notNull(),
  rawText: text("raw_text").notNull(),
  confidence: doublePrecision("confidence").notNull(),
  gate: text("gate").notNull(),
  timestamp: text("timestamp").notNull(),
  reason: text("reason").notNull(),
  status: text("status").default("Pending").notNull(),
  createdAt: timestamp("created_at").defaultNow().notNull(),
});

export type ReviewItemModel = typeof reviewQueueTable.$inferSelect;
export type InsertReviewItem = typeof reviewQueueTable.$inferInsert;

export const manualCorrectionsTable = pgTable("manual_corrections", {
  id: serial("id").primaryKey(),
  reviewId: integer("review_id").notNull(),
  originalText: text("original_text").notNull(),
  correctedText: text("corrected_text").notNull(),
  correctedBy: text("corrected_by").notNull(),
  correctedAt: timestamp("corrected_at").defaultNow().notNull(),
});

export type ManualCorrectionModel = typeof manualCorrectionsTable.$inferSelect;
export type InsertManualCorrection = typeof manualCorrectionsTable.$inferInsert;