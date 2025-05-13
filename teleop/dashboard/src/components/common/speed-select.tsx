import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Slider } from "@/components/ui/slider";
import { Gauge } from "lucide-react";
import { useState } from "react";

interface SpeedSelectProps {
  onChange?: (speed: number) => void;
  defaultValue?: number;
  disabled?: boolean;
  title?: string;
}

export function SpeedSelect({
  onChange,
  defaultValue = 1.0,
  disabled = false,
  title = "Playback Speed",
}: SpeedSelectProps) {
  const [speed, setSpeed] = useState(defaultValue);

  const handleSpeedChange = (value: number[]) => {
    const newSpeed = value[0];
    setSpeed(newSpeed);
    onChange?.(newSpeed);
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className="flex items-center gap-2 cursor-pointer"
          disabled={disabled}
        >
          <Gauge className="h-4 w-4" />
          <span>{speed.toFixed(1)}x</span>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-80">
        <div className="space-y-4">
          <h4 className="font-medium leading-none">{title}</h4>
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">0.1x</span>
            <span className="text-sm font-medium">{speed.toFixed(1)}x</span>
            <span className="text-sm text-muted-foreground">2.0x</span>
          </div>
          <Slider
            value={[speed]}
            min={0.1}
            max={2}
            step={0.1}
            onValueChange={handleSpeedChange}
          />
        </div>
      </PopoverContent>
    </Popover>
  );
}
