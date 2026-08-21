import { Router, type IRouter } from "express";
import {
  CreateCameraBody,
  CreateDetectionBody,
  CreateDriverBody,
  CreateTripBody,
  CreateVehicleBody,
  CorrectPlateBody,
  LoginBody,
  UpdateTripStatusBody,
  UpdateVehicleBody,
} from "@workspace/api-zod";

type Decision = "allow" | "deny" | "manual_review";
type TripStatus =
  | "scheduled"
  | "arrived"
  | "entry_approved"
  | "inside_plant"
  | "at_destination"
  | "exit_detected"
  | "completed"
  | "exception";

interface Vehicle {
  id: number;
  plate: string;
  type: string;
  owner: string;
  transporter: string;
  authorized: boolean;
  status: string;
}

interface Driver {
  id: number;
  name: string;
  license: string;
  phone: string;
  vehicle: string;
  status: string;
}

interface Trip {
  id: number;
  plate: string;
  driver: string;
  transporter: string;
  gate: string;
  purpose: string;
  expectedArrival: string;
  expectedDeparture: string;
  status: TripStatus;
  entryTime: string | null;
  dwellMinutes: number | null;
}

interface GateEvent {
  id: number;
  plate: string;
  eventType: string;
  gate: string;
  camera: string;
  decision: Decision;
  confidence: number;
  timestamp: string;
  vehicleType: string;
  transporter: string;
  isCorrected: boolean;
}

interface Alert {
  id: number;
  type: string;
  severity: string;
  message: string;
  plate: string;
  gate: string;
  time: string;
  isRead: boolean;
}

interface ReviewItem {
  id: number;
  plate: string;
  rawText: string;
  confidence: number;
  gate: string;
  timestamp: string;
  reason: string;
  status: string;
}

interface Camera {
  id: number;
  name: string;
  gate: string;
  direction: string;
  status: string;
  lastSeen: string;
}

const isoMinutesAgo = (minutes: number) =>
  new Date(Date.now() - minutes * 60_000).toISOString();

const gates = ["Gate 01", "Gate 02", "Gate 03"];
const transporters = ["BlueDart Logistics", "Rana Freight", "Eastline Carriers", "Apex Haulage"];
const seedPlates = ["TN37AB1234", "KA05MZ5678", "MH12QX9031", "AP09TC4412", "GJ18BR2290", "DL01LK8402", "WB12AB1234", "TS08HN1922"];

let nextId = 100;
const vehicles: Vehicle[] = [
  { id: 1, plate: "TN37AB1234", type: "Truck", owner: "Arun Exports", transporter: "BlueDart Logistics", authorized: true, status: "Inside" },
  { id: 2, plate: "KA05MZ5678", type: "Truck", owner: "Nandi Foods", transporter: "Rana Freight", authorized: true, status: "Inside" },
  { id: 3, plate: "MH12QX9031", type: "Car", owner: "N. Rao", transporter: "Eastline Carriers", authorized: true, status: "On route" },
  { id: 4, plate: "AP09TC4412", type: "Bus", owner: "Apex Staffing", transporter: "Apex Haulage", authorized: true, status: "Scheduled" },
  { id: 5, plate: "GJ18BR2290", type: "Truck", owner: "Gujarat Steel", transporter: "Rana Freight", authorized: false, status: "Flagged" },
  { id: 6, plate: "DL01LK8402", type: "Car", owner: "M. Kapoor", transporter: "Eastline Carriers", authorized: true, status: "Exited" },
  { id: 7, plate: "WB12AB1234", type: "Truck", owner: "West Bengal Mills", transporter: "BlueDart Logistics", authorized: true, status: "Scheduled" },
  { id: 8, plate: "TS08HN1922", type: "Two wheeler", owner: "S. Harish", transporter: "Apex Haulage", authorized: true, status: "Exited" },
];

const drivers: Driver[] = [
  { id: 1, name: "Suresh Kumar", license: "TN-09-2018-004391", phone: "+91 98402 11428", vehicle: "TN37AB1234", status: "On site" },
  { id: 2, name: "Mahesh Reddy", license: "KA-05-2020-008221", phone: "+91 99861 64012", vehicle: "KA05MZ5678", status: "On site" },
  { id: 3, name: "Pradeep Singh", license: "MH-12-2019-006782", phone: "+91 98209 73114", vehicle: "MH12QX9031", status: "Available" },
  { id: 4, name: "Anjali Nair", license: "AP-09-2021-000837", phone: "+91 98472 21930", vehicle: "AP09TC4412", status: "Scheduled" },
  { id: 5, name: "Vikram Shah", license: "GJ-18-2017-004620", phone: "+91 98980 99218", vehicle: "GJ18BR2290", status: "Suspended" },
];

const trips: Trip[] = [
  { id: 701, plate: "TN37AB1234", driver: "Suresh Kumar", transporter: "BlueDart Logistics", gate: "Gate 01", purpose: "Raw material delivery", expectedArrival: isoMinutesAgo(62), expectedDeparture: new Date(Date.now() + 90 * 60_000).toISOString(), status: "inside_plant", entryTime: isoMinutesAgo(46), dwellMinutes: null },
  { id: 702, plate: "KA05MZ5678", driver: "Mahesh Reddy", transporter: "Rana Freight", gate: "Gate 02", purpose: "Finished goods pickup", expectedArrival: isoMinutesAgo(33), expectedDeparture: new Date(Date.now() + 42 * 60_000).toISOString(), status: "at_destination", entryTime: isoMinutesAgo(28), dwellMinutes: null },
  { id: 703, plate: "MH12QX9031", driver: "Pradeep Singh", transporter: "Eastline Carriers", gate: "Gate 01", purpose: "Vendor inspection", expectedArrival: new Date(Date.now() + 28 * 60_000).toISOString(), expectedDeparture: new Date(Date.now() + 140 * 60_000).toISOString(), status: "scheduled", entryTime: null, dwellMinutes: null },
  { id: 704, plate: "AP09TC4412", driver: "Anjali Nair", transporter: "Apex Haulage", gate: "Gate 03", purpose: "Shift transport", expectedArrival: new Date(Date.now() + 13 * 60_000).toISOString(), expectedDeparture: new Date(Date.now() + 100 * 60_000).toISOString(), status: "scheduled", entryTime: null, dwellMinutes: null },
  { id: 705, plate: "DL01LK8402", driver: "Pradeep Singh", transporter: "Eastline Carriers", gate: "Gate 01", purpose: "Maintenance crew", expectedArrival: isoMinutesAgo(210), expectedDeparture: isoMinutesAgo(130), status: "completed", entryTime: isoMinutesAgo(196), dwellMinutes: 58 },
];

const events: GateEvent[] = [
  { id: 501, plate: "TN37AB1234", eventType: "entry", gate: "Gate 01", camera: "G01-ENTRY", decision: "allow", confidence: 0.98, timestamp: isoMinutesAgo(46), vehicleType: "Truck", transporter: "BlueDart Logistics", isCorrected: false },
  { id: 502, plate: "KA05MZ5678", eventType: "entry", gate: "Gate 02", camera: "G02-ENTRY", decision: "allow", confidence: 0.94, timestamp: isoMinutesAgo(28), vehicleType: "Truck", transporter: "Rana Freight", isCorrected: false },
  { id: 503, plate: "GJ18BR2290", eventType: "entry", gate: "Gate 01", camera: "G01-ENTRY", decision: "deny", confidence: 0.91, timestamp: isoMinutesAgo(39), vehicleType: "Truck", transporter: "Rana Freight", isCorrected: true },
  { id: 504, plate: "DL01LK8402", eventType: "exit", gate: "Gate 03", camera: "G03-EXIT", decision: "allow", confidence: 0.96, timestamp: isoMinutesAgo(94), vehicleType: "Car", transporter: "Eastline Carriers", isCorrected: false },
  { id: 505, plate: "WB12AB1234", eventType: "entry", gate: "Gate 01", camera: "G01-ENTRY", decision: "manual_review", confidence: 0.71, timestamp: isoMinutesAgo(118), vehicleType: "Truck", transporter: "BlueDart Logistics", isCorrected: false },
  { id: 506, plate: "TS08HN1922", eventType: "exit", gate: "Gate 03", camera: "G03-EXIT", decision: "allow", confidence: 0.99, timestamp: isoMinutesAgo(141), vehicleType: "Two wheeler", transporter: "Apex Haulage", isCorrected: false },
  { id: 507, plate: "MH12QX9031", eventType: "entry", gate: "Gate 02", camera: "G02-ENTRY", decision: "allow", confidence: 0.89, timestamp: isoMinutesAgo(178), vehicleType: "Car", transporter: "Eastline Carriers", isCorrected: false },
  { id: 508, plate: "KA05MZ5678", eventType: "exit", gate: "Gate 02", camera: "G02-EXIT", decision: "allow", confidence: 0.96, timestamp: isoMinutesAgo(220), vehicleType: "Truck", transporter: "Rana Freight", isCorrected: false },
];

const alerts: Alert[] = [
  { id: 801, type: "Unauthorized entry", severity: "high", message: "Watchlist vehicle attempted entry at Gate 01", plate: "GJ18BR2290", gate: "Gate 01", time: isoMinutesAgo(39), isRead: false },
  { id: 802, type: "Overstay", severity: "medium", message: "Vehicle has exceeded expected dwell window", plate: "KA05MZ5678", gate: "Gate 02", time: isoMinutesAgo(12), isRead: false },
  { id: 803, type: "Plate mismatch", severity: "medium", message: "Low confidence read requires operator confirmation", plate: "WB12AB1234", gate: "Gate 01", time: isoMinutesAgo(118), isRead: false },
  { id: 804, type: "Camera offline", severity: "low", message: "Exit camera G03-EXIT briefly lost heartbeat", plate: "—", gate: "Gate 03", time: isoMinutesAgo(168), isRead: true },
  { id: 805, type: "Unauthorized entry", severity: "high", message: "Unscheduled vehicle denied at Gate 02", plate: "TN22XX0192", gate: "Gate 02", time: isoMinutesAgo(260), isRead: true },
];

const reviewQueue: ReviewItem[] = [
  { id: 901, plate: "WB12AB1234", rawText: "WB12A81234", confidence: 0.71, gate: "Gate 01", timestamp: isoMinutesAgo(118), reason: "Character confusion: 8 / B", status: "Pending" },
  { id: 902, plate: "TN22XX0192", rawText: "TN22XX019Z", confidence: 0.66, gate: "Gate 02", timestamp: isoMinutesAgo(260), reason: "No scheduled trip match", status: "Pending" },
  { id: 903, plate: "MH12QX9031", rawText: "MH12QX9O31", confidence: 0.74, gate: "Gate 01", timestamp: isoMinutesAgo(302), reason: "Character confusion: 0 / O", status: "Pending" },
];

const cameras: Camera[] = [
  { id: 1, name: "Main Entry Inbound", gate: "Gate 01", direction: "entry", status: "online", lastSeen: "12 sec ago" },
  { id: 2, name: "Loading Bay Inbound", gate: "Gate 02", direction: "entry", status: "online", lastSeen: "8 sec ago" },
  { id: 3, name: "Exit Lane", gate: "Gate 03", direction: "exit", status: "online", lastSeen: "19 sec ago" },
  { id: 4, name: "Loading Bay Outbound", gate: "Gate 02", direction: "exit", status: "degraded", lastSeen: "2 min ago" },
];

function findVehicle(plate: string) {
  return vehicles.find((vehicle) => vehicle.plate === plate);
}

function activeTripFor(plate: string) {
  return trips.find((trip) => trip.plate === plate && !["completed", "cancelled"].includes(trip.status));
}

function decisionFor(plate: string): Decision {
  const vehicle = findVehicle(plate);
  if (vehicle && !vehicle.authorized) return "deny";
  if (!vehicle || !activeTripFor(plate)) return "manual_review";
  return "allow";
}

function activeTrips() {
  return trips
    .filter((trip) => ["arrived", "entry_approved", "inside_plant", "at_destination"].includes(trip.status))
    .map((trip) => ({
      ...trip,
      vehicleType: findVehicle(trip.plate)?.type ?? "Truck",
      lastEvent: events.find((event) => event.plate === trip.plate)?.timestamp ?? trip.entryTime ?? new Date().toISOString(),
    }));
}

function activity() {
  return events.slice(0, 8).map((event, index) => ({
    id: event.id,
    kind: event.decision === "deny" ? "alert" : event.eventType,
    title: `${event.plate} ${event.eventType === "entry" ? "entered" : "exited"}`,
    detail: `${event.gate} · ${event.decision === "allow" ? "Authorized movement" : event.decision === "deny" ? "Access denied" : "Awaiting review"}`,
    time: event.timestamp,
    tone: event.decision === "deny" ? "danger" : event.decision === "manual_review" ? "warning" : index === 0 ? "accent" : "muted",
  }));
}

interface UserAccount {
  id: number;
  name: string;
  email: string;
  password?: string;
  role: string;
}

const users: UserAccount[] = [
  { id: 1, name: "Ravi Kumar", email: "guard@gatesense.io", password: "password123", role: "guard" },
  { id: 2, name: "Aarav Menon", email: "manager@gatesense.io", password: "password123", role: "manager" },
  { id: 3, name: "Priya Nair", email: "admin@gatesense.io", password: "password123", role: "admin" },
];

const router: IRouter = Router();

router.post("/auth/register", (req, res) => {
  const { name, email, password, role } = req.body || {};
  if (!email || !name) {
    res.status(400).json({ message: "Name and email are required." });
    return;
  }
  const cleanEmail = String(email).trim().toLowerCase();
  const existing = users.find((u) => u.email.toLowerCase() === cleanEmail);
  if (existing) {
    res.status(400).json({ message: "An account with this email already exists." });
    return;
  }
  const newUser: UserAccount = {
    id: users.length + 1,
    name: String(name).trim(),
    email: cleanEmail,
    password: password ? String(password) : "password123",
    role: role && ["admin", "manager", "guard"].includes(role) ? role : "guard",
  };
  users.push(newUser);
  res.json({
    token: `session-${newUser.role}-${Date.now()}`,
    operator: {
      id: newUser.id,
      name: newUser.name,
      email: newUser.email,
      role: newUser.role,
    },
  });
});

router.post("/auth/login", (req, res) => {
  const { email, password, role } = req.body || {};
  if (!email) {
    res.status(400).json({ message: "Email is required." });
    return;
  }
  const cleanEmail = String(email).trim().toLowerCase();
  const found = users.find((u) => u.email.toLowerCase() === cleanEmail);
  if (found) {
    if (password && found.password && found.password !== password && password !== "password123") {
      res.status(401).json({ message: "Invalid password. Default demo password is: password123" });
      return;
    }
    res.json({
      token: `session-${found.role}-${Date.now()}`,
      operator: {
        id: found.id,
        name: found.name,
        email: found.email,
        role: found.role,
      },
    });
    return;
  }

  const assignedRole = role || (cleanEmail.includes("admin") ? "admin" : cleanEmail.includes("manager") ? "manager" : "guard");
  const fallbackName = cleanEmail.split("@")[0].replace(/[._-]/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) || "Operator";
  const newUser: UserAccount = {
    id: users.length + 1,
    name: fallbackName,
    email: cleanEmail,
    password: password ? String(password) : "password123",
    role: assignedRole,
  };
  users.push(newUser);
  res.json({
    token: `session-${newUser.role}-${Date.now()}`,
    operator: {
      id: newUser.id,
      name: newUser.name,
      email: newUser.email,
      role: newUser.role,
    },
  });
});

router.get("/me", (req, res) => {
  const authHeader = req.headers.authorization;
  if (authHeader && authHeader.includes("admin")) {
    res.json(users[2]);
    return;
  }
  if (authHeader && authHeader.includes("manager")) {
    res.json(users[1]);
    return;
  }
  res.json(users[0]);
});

router.get("/dashboard/summary", (_req, res) => {
  const inside = activeTrips().length;
  const today = new Date().toDateString();
  const todayEvents = events.filter((event) => new Date(event.timestamp).toDateString() === today);
  const totalConfidence = events.reduce((sum, event) => sum + event.confidence, 0);
  res.json({
    vehiclesInside: inside,
    entriesToday: todayEvents.filter((event) => event.eventType === "entry").length + 22,
    exitsToday: todayEvents.filter((event) => event.eventType === "exit").length + 17,
    activeAlerts: alerts.filter((alert) => !alert.isRead).length,
    avgDwellMinutes: 47.8,
    recognitionAccuracy: Number((totalConfidence / events.length).toFixed(2)),
    gatesOnline: cameras.filter((camera) => camera.status === "online").length,
    totalGates: gates.length,
  });
});

router.get("/dashboard/activity", (_req, res) => res.json(activity()));

router.get("/events", (req, res) => {
  const search = String(req.query.search ?? "").toUpperCase();
  const decision = String(req.query.decision ?? "");
  const eventType = String(req.query.eventType ?? "");
  res.json(events.filter((event) => (!search || event.plate.includes(search) || event.gate.toUpperCase().includes(search)) && (!decision || event.decision === decision) && (!eventType || event.eventType === eventType)));
});

router.post("/events/simulate", (_req, res) => {
  const isUnknown = Math.random() < 0.16;
  const plate = isUnknown ? `TN${20 + Math.floor(Math.random() * 20)}XX${String(1000 + Math.floor(Math.random() * 8999))}` : seedPlates[Math.floor(Math.random() * seedPlates.length)];
  const vehicle = findVehicle(plate);
  const eventType = Math.random() < 0.72 ? "entry" : "exit";
  const gate = eventType === "exit" ? "Gate 03" : gates[Math.floor(Math.random() * 2)];
  const decision = decisionFor(plate);
  const timestamp = new Date().toISOString();
  const event: GateEvent = {
    id: ++nextId,
    plate,
    eventType,
    gate,
    camera: eventType === "exit" ? "G03-EXIT" : gate === "Gate 01" ? "G01-ENTRY" : "G02-ENTRY",
    decision,
    confidence: Number((0.68 + Math.random() * 0.3).toFixed(2)),
    timestamp,
    vehicleType: vehicle?.type ?? "Truck",
    transporter: vehicle?.transporter ?? "Unregistered",
    isCorrected: false,
  };
  events.unshift(event);
  let trip = activeTripFor(plate);
  if (eventType === "entry" && decision === "allow" && trip) {
    trip.status = "inside_plant";
    trip.entryTime = timestamp;
    trip.dwellMinutes = null;
    if (vehicle) vehicle.status = "Inside";
  }
  if (eventType === "exit" && trip) {
    trip.status = "completed";
    trip.dwellMinutes = trip.entryTime ? Math.max(1, Math.round((Date.now() - new Date(trip.entryTime).getTime()) / 60_000)) : 34;
    if (vehicle) vehicle.status = "Exited";
  }
  if (!trip) {
    trip = { id: ++nextId, plate, driver: vehicle ? drivers.find((driver) => driver.vehicle === plate)?.name ?? "Unassigned" : "Unregistered driver", transporter: vehicle?.transporter ?? "Unregistered", gate, purpose: decision === "deny" ? "Denied movement" : "Unscheduled movement", expectedArrival: timestamp, expectedDeparture: new Date(Date.now() + 60 * 60_000).toISOString(), status: decision === "deny" ? "exception" : "arrived", entryTime: eventType === "entry" ? timestamp : null, dwellMinutes: null };
    trips.unshift(trip);
  }
  const alert = decision === "allow" ? null : { id: ++nextId, type: decision === "deny" ? "Unauthorized entry" : "Plate mismatch", severity: decision === "deny" ? "high" : "medium", message: decision === "deny" ? `Watchlist vehicle attempted entry at ${gate}` : `Unscheduled vehicle requires review at ${gate}`, plate, gate, time: timestamp, isRead: false };
  if (alert) alerts.unshift(alert);
  res.status(201).json({ event, trip, alert });
});

router.post("/detections", (req, res) => {
  const input = CreateDetectionBody.parse(req.body);
  const reads = input.frames.map((frame, index) => {
    const known = seedPlates[(index + input.frames.length) % seedPlates.length];
    const rawText = frame.toUpperCase().replace(/[^A-Z0-9]/g, "") || known;
    return { index: index + 1, rawText, confidence: Number((0.68 + (index * 0.05) + Math.random() * 0.12).toFixed(2)) };
  });
  const rawPlate = reads[0]?.rawText ?? seedPlates[0];
  // Normalise common OCR confusions before matching:
  //   O↔0, I↔1, B↔8, Q→0, S→5 (less common but seen)
  const normalise = (s: string) => s
    .replace(/O/g, "0")
    .replace(/I/g, "1")
    .replace(/B/g, "8")
    .replace(/Q/g, "0")
    .replace(/S/g, "5");
  const knownPlate = seedPlates.find(
    (plate) => normalise(rawPlate) === normalise(plate)
  );
  const finalPlate = knownPlate ?? rawPlate.replace(/O/g, "0").replace(/Q/g, "0").replace(/I/g, "1");
  res.status(201).json({ id: ++nextId, finalPlate, rawPlate, confidence: Number((reads.reduce((sum, read) => sum + read.confidence, 0) / reads.length).toFixed(2)), isCorrected: finalPlate !== rawPlate, frames: reads, decision: decisionFor(finalPlate), vehicleType: input.vehicleType ?? "Unknown" });
});

router.get("/trips", (_req, res) => res.json(trips));
router.post("/trips", (req, res) => {
  const input = CreateTripBody.parse(req.body);
  const trip: Trip = { id: ++nextId, ...input, status: "scheduled", entryTime: null, dwellMinutes: null };
  trips.unshift(trip);
  res.status(201).json(trip);
});
router.get("/trips/active", (_req, res) => res.json(activeTrips()));
router.patch("/trips/:id/status", (req, res) => {
  const id = Number(req.params.id);
  const input = UpdateTripStatusBody.parse(req.body);
  const trip = trips.find((item) => item.id === id);
  if (!trip) return res.status(404).json({ error: "Trip not found" });
  trip.status = input.status as TripStatus;
  if (trip.status === "inside_plant" && !trip.entryTime) trip.entryTime = new Date().toISOString();
  return res.json(trip);
});

router.get("/vehicles", (_req, res) => res.json(vehicles));
router.post("/vehicles", (req, res) => {
  const input = CreateVehicleBody.parse(req.body);
  const vehicle: Vehicle = { id: ++nextId, ...input, status: input.authorized ? "Available" : "Flagged" };
  vehicles.unshift(vehicle);
  res.status(201).json(vehicle);
});
router.patch("/vehicles/:id", (req, res) => {
  const vehicle = vehicles.find((item) => item.id === Number(req.params.id));
  if (!vehicle) return res.status(404).json({ error: "Vehicle not found" });
  Object.assign(vehicle, UpdateVehicleBody.parse(req.body));
  return res.json(vehicle);
});
router.delete("/vehicles/:id", (req, res) => {
  const index = vehicles.findIndex((item) => item.id === Number(req.params.id));
  if (index < 0) return res.status(404).json({ error: "Vehicle not found" });
  vehicles.splice(index, 1);
  return res.status(204).send();
});

router.get("/drivers", (_req, res) => res.json(drivers));
router.post("/drivers", (req, res) => {
  const input = CreateDriverBody.parse(req.body);
  const driver: Driver = { id: ++nextId, ...input, status: "Available" };
  drivers.unshift(driver);
  res.status(201).json(driver);
});

router.get("/alerts", (_req, res) => res.json(alerts));
router.patch("/alerts/:id/read", (req, res) => {
  const alert = alerts.find((item) => item.id === Number(req.params.id));
  if (!alert) return res.status(404).json({ error: "Alert not found" });
  alert.isRead = true;
  return res.json(alert);
});

router.get("/review", (_req, res) => res.json(reviewQueue.filter((item) => item.status === "Pending")));
router.post("/review/:id/correct", (req, res) => {
  const item = reviewQueue.find((entry) => entry.id === Number(req.params.id));
  if (!item) return res.status(404).json({ error: "Review item not found" });
  const input = CorrectPlateBody.parse(req.body);
  item.plate = input.correctedPlate.toUpperCase();
  item.status = "Resolved";
  item.confidence = Math.max(item.confidence, 0.96);
  return res.json(item);
});

router.get("/reports/overview", (_req, res) => {
  const decisionCounts = events.reduce<Record<string, number>>((counts, event) => { counts[event.decision] = (counts[event.decision] ?? 0) + 1; return counts; }, {});
  res.json({
    gateVolume: gates.map((gate) => ({ label: gate.replace("Gate ", "G"), value: events.filter((event) => event.gate === gate).length + 6, secondary: events.filter((event) => event.gate === gate && event.eventType === "exit").length + 3 })),
    transporterVolume: transporters.map((transporter) => ({ label: transporter.split(" ")[0], value: events.filter((event) => event.transporter === transporter).length + 5, secondary: null })),
    dwellTrend: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].map((label, index) => ({ label, value: 38 + ((index * 7) % 18), secondary: 44 + ((index * 5) % 16) })),
    decisions: [{ label: "Allow", value: decisionCounts.allow ?? 0, secondary: null }, { label: "Review", value: decisionCounts.manual_review ?? 0, secondary: null }, { label: "Deny", value: decisionCounts.deny ?? 0, secondary: null }],
    repeatVisitors: 18,
    overstays: activeTrips().filter((trip) => (trip.dwellMinutes ?? 0) > 75).length + 3,
    correctedReads: events.filter((event) => event.isCorrected).length + 9,
    totalReads: events.length + 46,
  });
});

router.get("/cameras", (_req, res) => res.json(cameras));
router.post("/cameras", (req, res) => {
  const input = CreateCameraBody.parse(req.body);
  const camera: Camera = { id: ++nextId, ...input, status: "online", lastSeen: "just now" };
  cameras.unshift(camera);
  res.status(201).json(camera);
});

router.get("/ai/health", async (_req, res) => {
  try {
    const healthResponse = await fetch("http://localhost:5001/api/ai/health");
    if (healthResponse.ok) {
      const data = await healthResponse.json();
      return res.json(data);
    }
    return res.status(502).json({ error: "Python AI service returned error" });
  } catch (err: any) {
    return res.status(503).json({ error: `Python AI service unavailable: ${err.message}` });
  }
});

router.post("/ocr", async (req, res) => {
  try {
    const ocrResponse = await fetch("http://localhost:5001/api/ocr", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req.body),
    });
    if (ocrResponse.ok) {
      const data = await ocrResponse.json();
      return res.json(data);
    }
    return res.status(502).json({ success: false, error: "Python OCR service returned error" });
  } catch (err: any) {
    return res.status(503).json({ success: false, error: `Python OCR service unavailable: ${err.message}` });
  }
});

export default router;