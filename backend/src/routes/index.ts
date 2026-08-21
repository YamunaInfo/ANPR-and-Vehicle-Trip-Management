import { Router, type IRouter } from "express";
import healthRouter from "./health";
import gatesenseRouter from "./gatesense";

const router: IRouter = Router();

router.use(healthRouter);
router.use(gatesenseRouter);

export default router;
