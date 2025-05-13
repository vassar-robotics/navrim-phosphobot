import controlschema from "@/assets/ControlSchema.png";
import { LoadingPage } from "@/components/common/loading";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fetcher } from "@/lib/utils";
import { ServerStatus } from "@/types";
import {
  ArrowDown,
  ArrowDownFromLine,
  ArrowLeft,
  ArrowRight,
  ArrowUp,
  ArrowUpFromLine,
  ChevronDown,
  ChevronUp,
  Play,
  RotateCcw,
  RotateCw,
  Space,
  Square,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import useSWR from "swr";

export default function RobotControllerPage() {
  const { data: serverStatus, error: serverError } = useSWR<ServerStatus>(
    ["/status"],
    fetcher,
    {
      refreshInterval: 5000,
    },
  );

  const [isMoving, setIsMoving] = useState(false);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [selectedRobotName, setSelectedRobotName] = useState<string | null>(
    null,
  );

  // Refs to manage our control loop and state
  const keysPressedRef = useRef(new Set<string>());
  const intervalIdRef = useRef<NodeJS.Timeout | null>(null);
  const lastExecutionTimeRef = useRef(0);
  const openStateRef = useRef(1);

  // Configuration constants (from control.js)
  const BASE_URL = `http://${window.location.hostname}:${window.location.port}/`;
  const STEP_SIZE = 1; // in centimeters
  const LOOP_INTERVAL = 10; // ms (~50 Hz)
  const INSTRUCTIONS_PER_SECOND = 30;
  const DEBOUNCE_INTERVAL = 1000 / INSTRUCTIONS_PER_SECOND;

  interface RobotMovement {
    x: number;
    y: number;
    z: number;
    rz: number;
    rx: number;
    ry: number;
    toggleOpen?: boolean;
  }

  // Mappings for keys (from control.js)
  const KEY_MAPPINGS: Record<string, RobotMovement> = {
    a: { x: 0, y: 0, z: STEP_SIZE, rz: 0, rx: 0, ry: 0 },
    d: { x: 0, y: 0, z: -STEP_SIZE, rz: 0, rx: 0, ry: 0 },
    ArrowUp: { x: STEP_SIZE, y: 0, z: 0, rz: 0, rx: 0, ry: 0 },
    ArrowDown: { x: -STEP_SIZE, y: 0, z: 0, rz: 0, rx: 0, ry: 0 },
    ArrowRight: { x: 0, y: 0, z: 0, rz: -STEP_SIZE * 3.14, rx: 0, ry: 0 },
    ArrowLeft: { x: 0, y: 0, z: 0, rz: STEP_SIZE * 3.14, rx: 0, ry: 0 },
    q: { x: 0, y: 0, z: 0, rz: 0, rx: STEP_SIZE * 3.14, ry: 0 },
    e: { x: 0, y: 0, z: 0, rz: 0, rx: -STEP_SIZE * 3.14, ry: 0 },
    w: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: STEP_SIZE * 3.14 },
    s: { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: -STEP_SIZE * 3.14 },
    " ": { x: 0, y: 0, z: 0, rz: 0, rx: 0, ry: 0, toggleOpen: true },
  };

  const robotIDFromName = (name?: string | null) => {
    if (name === undefined || name === null || !serverStatus?.robot_status) {
      return 0; // Default to the first robot
    }
    // Return the index in the robot_status array
    return serverStatus?.robot_status.findIndex(
      (robot) => robot.usb_port === name,
    );
  };

  // Utility: send POST request with JSON data
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const postData = async (url: string, data: any, queryParam?: any) => {
    try {
      let newUrl = url;
      if (queryParam) {
        const urlParams = new URLSearchParams(queryParam);
        if (urlParams.toString()) {
          newUrl += "?" + urlParams.toString();
        }
      }

      const response = await fetch(newUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        throw new Error("Network response was not ok");
      }
      return await response.json();
    } catch (error) {
      console.error("Error:", error);
    }
  };

  // Setup keyboard event listeners (integrating the logic from control.js)
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat) return;
      setActiveKey(event.key);

      if (KEY_MAPPINGS[event.key.toLowerCase()]) {
        keysPressedRef.current.add(event.key.toLowerCase());
      } else if (KEY_MAPPINGS[event.key]) {
        keysPressedRef.current.add(event.key);
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      setActiveKey(null);
      if (KEY_MAPPINGS[event.key.toLowerCase()]) {
        keysPressedRef.current.delete(event.key.toLowerCase());
      } else if (KEY_MAPPINGS[event.key]) {
        keysPressedRef.current.delete(event.key);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, []);

  useEffect(() => {
    if (
      !selectedRobotName &&
      serverStatus?.robot_status &&
      serverStatus.robot_status.length > 0 &&
      serverStatus.robot_status[0].usb_port
    ) {
      setSelectedRobotName(serverStatus.robot_status[0].usb_port);
    }
  }, [
    serverStatus,
    selectedRobotName,
    JSON.stringify(serverStatus?.robot_status), // eslint-disable-line react-hooks/exhaustive-deps
  ]);

  // Start the control loop only when the robot is moving.
  useEffect(() => {
    if (isMoving) {
      // Control loop: aggregate pressed keys and send movement commands.
      const controlRobot = () => {
        const currentTime = Date.now();
        if (currentTime - lastExecutionTimeRef.current >= DEBOUNCE_INTERVAL) {
          let deltaX = 0,
            deltaY = 0,
            deltaZ = 0,
            deltaRZ = 0,
            deltaRX = 0,
            deltaRY = 0;
          let didToggleOpen = false;

          keysPressedRef.current.forEach((key) => {
            if (KEY_MAPPINGS[key]) {
              deltaX += KEY_MAPPINGS[key].x;
              deltaY += KEY_MAPPINGS[key].y;
              deltaZ += KEY_MAPPINGS[key].z;
              deltaRZ += KEY_MAPPINGS[key].rz;
              deltaRX += KEY_MAPPINGS[key].rx;
              deltaRY += KEY_MAPPINGS[key].ry;
              if (KEY_MAPPINGS[key].toggleOpen) {
                didToggleOpen = true;
              }
            }
          });

          if (didToggleOpen) {
            // flip the gripper open/closed
            openStateRef.current = openStateRef.current > 0.99 ? 0 : 1;
            // remove space from the pressed-keys set so it only toggles once per key-down
            keysPressedRef.current.delete(" ");
          }

          if (
            deltaX !== 0 ||
            deltaY !== 0 ||
            deltaZ !== 0 ||
            deltaRZ !== 0 ||
            deltaRX !== 0 ||
            deltaRY !== 0 ||
            didToggleOpen
          ) {
            const data = {
              x: deltaX,
              y: deltaY,
              z: deltaZ,
              rx: deltaRX,
              ry: deltaRY,
              rz: deltaRZ,
              open: openStateRef.current,
            };
            postData(BASE_URL + "move/relative", data, {
              robot_id: robotIDFromName(selectedRobotName),
            });
          }
          lastExecutionTimeRef.current = currentTime;
        }
      };

      const intervalId = setInterval(controlRobot, LOOP_INTERVAL);
      intervalIdRef.current = intervalId;
      return () => clearInterval(intervalId);
    }
  }, [isMoving]);

  // Initialize the robot (using the same sequence from control.js)
  const initRobot = async () => {
    try {
      await postData(
        BASE_URL + "move/init",
        {},
        {
          robot_id: robotIDFromName(selectedRobotName),
        },
      );
      // Wait 2 seconds before setting the absolute position
      await new Promise((resolve) => setTimeout(resolve, 2000));
      const initData = {
        x: 0,
        y: 0,
        z: 0,
        rx: 0,
        ry: 0,
        rz: 0,
        open: 1,
      };
      await postData(BASE_URL + "move/absolute", initData, {
        robot_id: robotIDFromName(selectedRobotName),
      });
    } catch (error) {
      console.error("Error during init:", error);
    }
  };

  // Handlers for start/stop moving
  const startMoving = async () => {
    await initRobot();
    setIsMoving(true);
  };

  const stopMoving = async () => {
    setIsMoving(false);
  };

  // UI control mapping for instructions display
  const controls = [
    {
      key: "ArrowUp",
      description: "Move in the positive X direction",
      icon: <ArrowUp className="size-6" />,
    },
    {
      key: "ArrowDown",
      description: "Move in the negative X direction",
      icon: <ArrowDown className="size-6" />,
    },
    {
      key: "ArrowRight",
      description: "Rotate Z clockwise (yaw)",
      icon: <ArrowRight className="size-6" />,
    },
    {
      key: "ArrowLeft",
      description: "Rotate Z counter-clockwise (yaw)",
      icon: <ArrowLeft className="size-6" />,
    },
    {
      key: "a",
      description: "Increase Z (move up)",
      icon: <ChevronUp className="size-6" />,
    },
    {
      key: "d",
      description: "Decrease Z (move down)",
      icon: <ChevronDown className="size-6" />,
    },
    {
      key: "q",
      description: "Wrist pitch up",
      icon: <ArrowUpFromLine className="size-6" />,
    },
    {
      key: "e",
      description: "Wrist pitch down",
      icon: <ArrowDownFromLine className="size-6" />,
    },
    {
      key: "w",
      description: "Wrist roll clockwise",
      icon: <RotateCw className="size-6" />,
    },
    {
      key: "s",
      description: "Wrist roll counter-clockwise",
      icon: <RotateCcw className="size-6" />,
    },
    {
      key: " ",
      description: "Toggle the open state",
      icon: <Space className="size-6" />,
    },
  ];

  // Loading state
  if (!serverStatus && !serverError) {
    return <LoadingPage />;
  }

  return (
    <div className="container mx-auto px-4 py-6 space-y-8">
      <Card>
        <CardContent>
          <figure className="flex flex-col items-center">
            <img
              src={controlschema}
              alt="Robot Control Schema"
              className="w-full max-w-md rounded-md shadow"
            />
            <figcaption className="text-sm text-muted-foreground text-center mt-2">
              Press the key to move the robot in the corresponding direction
            </figcaption>
          </figure>
          <div className="flex items-center justify-center mt-6 gap-x-2">
            <Select
              value={selectedRobotName || ""}
              onValueChange={(value) => setSelectedRobotName(value)}
              disabled={isMoving}
            >
              <SelectTrigger id="follower-robot">
                <SelectValue placeholder="Select robot to move" />
              </SelectTrigger>
              <SelectContent>
                {serverStatus?.robot_status.map((robot) => (
                  <SelectItem
                    key={robot.usb_port}
                    value={robot.usb_port || "Undefined port"}
                  >
                    {robot.name} ({robot.usb_port})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            {isMoving && (
              <Button variant="destructive" onClick={stopMoving}>
                <Square className="mr-2 h-4 w-4" />
                Stop the Robot
              </Button>
            )}
            {!isMoving && (
              <Button variant="default" onClick={startMoving}>
                <Play className="mr-2 h-4 w-4" />
                Start Moving Robot
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {controls.map((control) => (
          <TooltipProvider key={control.key}>
            <Tooltip>
              <TooltipTrigger asChild>
                <Card
                  className={`flex flex-col items-center justify-center p-4 cursor-pointer hover:bg-accent transition-colors ${
                    activeKey === control.key
                      ? "bg-primary text-primary-foreground"
                      : ""
                  }`}
                  onClick={() => {
                    // Simulate key press
                    // const event = {
                    //   key: control.key,
                    //   repeat: false,
                    // } as KeyboardEvent;

                    // Use the same logic as in the keyDown handler
                    if (control.key === " ") {
                      // Toggle the open state and send a zero-movement command.
                      openStateRef.current = openStateRef.current === 1 ? 0 : 1;
                      const data = {
                        x: 0,
                        y: 0,
                        z: 0,
                        rx: 0,
                        ry: 0,
                        rz: 0,
                        open: openStateRef.current,
                      };
                      postData(BASE_URL + "move/relative", data, {
                        robot_id: robotIDFromName(selectedRobotName),
                      });
                    } else if (KEY_MAPPINGS[control.key.toLowerCase()]) {
                      keysPressedRef.current.add(control.key.toLowerCase());
                      setActiveKey(control.key);

                      // Set a timeout to clear the key press after a short delay
                      setTimeout(() => {
                        keysPressedRef.current.delete(
                          control.key.toLowerCase(),
                        );
                        setActiveKey(null);
                      }, 300);
                    } else if (KEY_MAPPINGS[control.key]) {
                      keysPressedRef.current.add(control.key);
                      setActiveKey(control.key);

                      // Set a timeout to clear the key press after a short delay
                      setTimeout(() => {
                        keysPressedRef.current.delete(control.key);
                        setActiveKey(null);
                      }, 300);
                    }
                  }}
                >
                  {control.icon}
                  <span className="mt-2 font-bold">
                    {control.key === " " ? "Space" : control.key.toUpperCase()}
                  </span>
                </Card>
              </TooltipTrigger>
              <TooltipContent>
                <p>{control.description}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        ))}
      </div>
    </div>
  );
}
