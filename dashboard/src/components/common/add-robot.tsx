import placeholderSvg from "@/assets/placeholder.svg";
import { AutoComplete } from "@/components/common/autocomplete";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { Loader2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";
import useSWR from "swr";

// Data model for robot types
const ROBOT_TYPES = [
  {
    id: "go2",
    name: "Unitree Go2",
    category: "mobile",
    image: placeholderSvg,
    fields: [{ name: "ip_address", label: "IP Address", type: "ip" }],
  },
  {
    id: "lekiwi",
    name: "LeKiwi",
    category: "mobile",
    image: placeholderSvg,
    fields: [
      { name: "ip_address", label: "IP Address", type: "ip" },
      { name: "port", label: "Port", type: "port" },
    ],
  },
  {
    id: "koch",
    name: "Koch 1.1",
    category: "manipulator",
    image: placeholderSvg,
    fields: [{ name: "usb_port", label: "USB Port", type: "usb_port" }],
  },
  {
    id: "so100",
    name: "SO-100 / SO-101",
    category: "manipulator",
    image: placeholderSvg,
    fields: [{ name: "usb_port", label: "USB Port", type: "usb_port" }],
  },
];

interface RobotConfigModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function RobotConfigModal({
  open,
  onOpenChange,
}: RobotConfigModalProps) {
  const [selectedRobotType, setSelectedRobotType] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [formValues, setFormValues] = useState<Record<string, any>>({});

  const selectedRobot = ROBOT_TYPES.find(
    (robot) => robot.id === selectedRobotType,
  );

  // Fetch IP addresses for autocomplete
  const { data: ipAddresses, isLoading: isLoadingIps } = useSWR(
    selectedRobot?.fields.some((f) => f.type === "ip")
      ? "/api/available-ips"
      : null,
    fetcher,
    {
      fallbackData: [
        { value: "192.168.1.100", label: "192.168.1.100 (Robot 1)" },
        { value: "192.168.1.101", label: "192.168.1.101 (Robot 2)" },
        { value: "10.0.0.50", label: "10.0.0.50 (Lab Robot)" },
      ],
    },
  );

  // Fetch USB ports for autocomplete
  const { data: usbPorts, isLoading: isLoadingUsb } = useSWR(
    selectedRobot?.fields.some((f) => f.type === "usb_port")
      ? "/api/available-usb-ports"
      : null,
    fetcher,
    {
      fallbackData: [
        { value: "/dev/ttyUSB0", label: "/dev/ttyUSB0" },
        { value: "/dev/ttyUSB1", label: "/dev/ttyUSB1" },
        { value: "COM3", label: "COM3 (Windows)" },
      ],
    },
  );

  const handleRobotTypeChange = (value: string) => {
    setSelectedRobotType(value);
    // Reset form values when robot type changes
    setFormValues({});
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleFieldChange = (fieldName: string, value: any) => {
    setFormValues((prev) => ({
      ...prev,
      [fieldName]: value,
    }));
  };

  const handleSubmit = async () => {
    if (!selectedRobot) return;

    // Check if all required fields are filled
    const missingFields = selectedRobot.fields.filter(
      (field) => !formValues[field.name],
    );

    if (missingFields.length > 0) {
      toast.error(
        `Please fill in all required fields: ${missingFields.map((f) => f.label).join(", ")}`,
      );
      return;
    }

    setIsSubmitting(true);

    try {
      // Prepare payload
      const payload = {
        robotType: selectedRobotType,
        ...formValues,
      };

      // Call API to add robot
      const response = await fetchWithBaseUrl(
        "/api/add-robot",
        "POST",
        payload,
      );

      if (response) {
        toast.success(
          `${selectedRobot.name} robot has been added successfully.`,
        );

        // Close modal on success
        onOpenChange(false);

        // Reset form
        setSelectedRobotType("");
        setFormValues({});
      }
    } catch (error) {
      console.error("Error adding robot:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[600px]">
        <DialogHeader>
          <DialogTitle>Add New Robot</DialogTitle>
          <DialogDescription>
            Configure a new robot to add to your system.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-6 py-2">
          <div className="grid grid-cols-[2fr_1fr] gap-4 items-start">
            <div className="space-y-2">
              <Label htmlFor="robot-type">Robot Type</Label>
              <Select
                value={selectedRobotType}
                onValueChange={handleRobotTypeChange}
              >
                <SelectTrigger id="robot-type">
                  <SelectValue placeholder="Select robot type" />
                </SelectTrigger>
                <SelectContent>
                  {ROBOT_TYPES.map((robot) => (
                    <SelectItem key={robot.id} value={robot.id}>
                      {robot.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedRobot && (
              <div className="flex flex-col items-center justify-center">
                <div className="relative h-[120px] w-[120px] rounded-md border overflow-hidden">
                  <img
                    src={selectedRobot.image || "/placeholder.svg"}
                    alt={selectedRobot.name}
                    className="object-cover w-[120px] h-[120px]"
                  />
                </div>
                <span className="text-xs text-muted-foreground mt-1">
                  {selectedRobot.category === "mobile"
                    ? "Mobile Unit"
                    : "Manipulator"}
                </span>
              </div>
            )}
          </div>

          {selectedRobot && (
            <div className="space-y-4">
              {selectedRobot.fields.map((field) => (
                <div key={field.name} className="space-y-2">
                  <Label htmlFor={field.name}>{field.label}</Label>

                  {field.type === "ip" && (
                    <AutoComplete
                      options={ipAddresses || []}
                      value={formValues[field.name]}
                      onValueChange={(value) =>
                        handleFieldChange(field.name, value)
                      }
                      isLoading={isLoadingIps}
                      placeholder="Select or enter IP address"
                      emptyMessage="No IP addresses found"
                      allowCustomValue={true}
                    />
                  )}

                  {field.type === "usb_port" && (
                    <AutoComplete
                      options={usbPorts || []}
                      value={formValues[field.name]}
                      onValueChange={(value) =>
                        handleFieldChange(field.name, value)
                      }
                      isLoading={isLoadingUsb}
                      placeholder="Select USB port"
                      emptyMessage="No USB ports detected"
                      allowCustomValue={true}
                    />
                  )}

                  {field.type === "port" && (
                    <Input
                      id={field.name}
                      type="number"
                      placeholder="Enter port number"
                      value={formValues[field.name] || ""}
                      onChange={(e) =>
                        handleFieldChange(field.name, e.target.value)
                      }
                    />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={!selectedRobot || isSubmitting}
            className="min-w-[120px]"
          >
            {isSubmitting ? (
              <>
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Adding...
              </>
            ) : (
              "Add Robot"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
