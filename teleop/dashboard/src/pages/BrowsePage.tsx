import { LoadingPage } from "@/components/common/loading";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useGlobalStore } from "@/lib/hooks";
import { fetchWithBaseUrl, fetcher } from "@/lib/utils";
import { ServerStatus } from "@/types";
import {
  AlertCircle,
  ArrowUpFromLine,
  ChevronRight,
  Download,
  ExternalLink,
  Eye,
  File,
  Folder,
  LoaderCircle,
  MoreVertical,
  Repeat,
  Trash2,
} from "lucide-react";
import { useMemo, useState } from "react";
import { redirect, useParams, useSearchParams } from "react-router-dom";
import { Link } from "react-router-dom";
import { toast } from "sonner";
import useSWR from "swr";

interface FileItem {
  name: string;
  path: string;
  absolute_path: string;
  is_dir: boolean;
  is_dataset_dir: boolean;
  browseUrl: string;
  huggingfaceUrl?: string;
  downloadUrl?: string;
  previewUrl?: string | null;
  canDeleteDataset?: boolean;
  deleteDatasetAction?: string;
}
interface BrowseData {
  directoryTitle: string;
  tokenError?: string;
  items: FileItem[];
  episode_ids?: number[];
  episode_paths?: string[];
}

// const handleRevealInFinder = (path: string) => {
//   // `item.path` should be the absolute path to the file or directory on the user's local system
//   // If it's a file, extract the containing directory; if it's a directory, use it directly
//   const directoryPath = path.endsWith("/")
//     ? path
//     : path.substring(0, path.lastIndexOf("/"));

//   // Construct the file:// URL for the directory
//   const fileUrl = `file://${directoryPath}`;

//   // Open the URL in a new tab/window, which may trigger the file explorer
//   window.open(fileUrl, "_blank");
// };

export default function FileBrowser() {
  const params = useParams();
  const [searchParams] = useSearchParams();
  const path = params.path || searchParams.get("path") || "";

  const { data, error, mutate } = useSWR<BrowseData>(
    ["/files", path],
    ([url]) => fetcher(url, "POST", { path }),
  );

  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [confirmEpisodeDeleteOpen, setConfirmEpisodeDeleteOpen] =
    useState(false);
  const [selectedItem, setSelectedItem] = useState<FileItem | null>(null);
  const [selectedEpisode, setSelectedEpisode] = useState("");
  const leaderArmSerialIds = useGlobalStore(
    (state) => state.leaderArmSerialIds,
  );

  // Loading state for episode deletion
  const [loadingDeleteEpisode, setLoadingDeleteEpisode] = useState(false);

  const { data: serverStatus } = useSWR<ServerStatus>(
    ["/status"],
    ([url]) => fetcher(url),
    {
      refreshInterval: 5000,
    },
  );

  const isRobotConnected = useMemo(() => {
    return serverStatus?.robots && serverStatus.robots.length > 0;
  }, [serverStatus]);

  if (error)
    return (
      <Alert variant="destructive">
        <AlertCircle className="h-4 w-4" />
        <AlertTitle>Error</AlertTitle>
        <AlertDescription>Failed to load data.</AlertDescription>
      </Alert>
    );

  if (!data) return <LoadingPage />;

  const handleDeleteDataset = (item: FileItem) => {
    setSelectedItem(item);
    setConfirmDeleteOpen(true);
  };

  const handleDeleteEpisode = async () => {
    // Ensure an episode is selected.
    if (!selectedEpisode) return;
    fetchWithBaseUrl(`/episode/delete`, "POST", {
      path: path,
      episode_id: selectedEpisode,
    }).then((response) => {
      if (response.status !== "ok") {
        toast.error("Failed to delete episode");
      } else {
        toast.success("Episode deleted successfully");
        mutate();
        redirect(path);
      }
    });
  };

  const handleReplayEpisode = async () => {
    if (!selectedEpisode || !data.episode_paths) return;
    const episode_path = data.episode_paths[parseInt(selectedEpisode)];
    if (!episode_path) return;
    const robot_serials_to_ignore = leaderArmSerialIds ?? null;
    fetchWithBaseUrl(`/recording/play`, "POST", {
      episode_path,
      robot_serials_to_ignore,
    });
  };

  const handleConfirmDelete = async () => {
    if (!selectedItem) return;

    try {
      const resp = await fetchWithBaseUrl(
        `/dataset/delete?path=${encodeURIComponent(selectedItem.path)}`,
        "POST",
      );

      if (resp.status !== "ok") {
        toast.error("Failed to delete dataset");
        return;
      }

      toast.success("Dataset deleted successfully");
      mutate();
      setConfirmDeleteOpen(false);
      setSelectedItem(null);

      // split on both "/" and "\\" so it works on Windows & POSIX
      const segments = selectedItem.path.split(/[/\\]+/).filter(Boolean);
      const parentSegments = segments.slice(0, -1);
      const parentPath = parentSegments.length
        ? parentSegments.join("/") // always join with "/" for URLs
        : "";

      if (parentPath) {
        redirect(parentPath);
      } else {
        redirect("/browse");
      }
    } catch (error) {
      console.error(error);
      toast.error("An unexpected error occurred");
    }
  };

  const handlePushToHub = async (item: FileItem) => {
    fetchWithBaseUrl(`/dataset/sync?path=${item.path}`, "POST");
  };

  const pathParts = path.split("/").filter(Boolean);

  return (
    <div className="container mx-auto py-6 px-4 md:px-6">
      <Breadcrumb className="mb-4">
        <BreadcrumbList>
          <BreadcrumbItem>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <BreadcrumbLink asChild>
                    <Link to="/browse">phosphobot</Link>
                  </BreadcrumbLink>
                </TooltipTrigger>
                <TooltipContent>
                  <p>
                    Explore your datasets in{" "}
                    <code>~/phosphobot/recordings</code>
                  </p>
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </BreadcrumbItem>
          <BreadcrumbSeparator>
            <ChevronRight className="h-4 w-4" />
          </BreadcrumbSeparator>
          {pathParts.map((part, index) => (
            <BreadcrumbItem key={index}>
              <BreadcrumbLink asChild>
                <Link
                  to={`/browse?path=${pathParts.slice(0, index + 1).join("/")}`}
                >
                  {part}
                </Link>
              </BreadcrumbLink>
              {index < pathParts.length - 1 && (
                <BreadcrumbSeparator>
                  <ChevronRight className="h-4 w-4" />
                </BreadcrumbSeparator>
              )}
            </BreadcrumbItem>
          ))}
        </BreadcrumbList>
      </Breadcrumb>

      {data.tokenError && (
        <Alert variant="destructive" className="mb-4">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{data.tokenError}</AlertDescription>
        </Alert>
      )}

      <Table className="bg-background rounded-lg">
        <TableHeader>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell></TableCell>
          </TableRow>
        </TableHeader>
        <TableBody>
          {data.items.map((item) => (
            <TableRow key={item.path}>
              <TableCell>
                <Link
                  to={item.browseUrl}
                  className="flex items-center text-blue-500 hover:underline"
                >
                  {item.is_dir ? (
                    <Folder className="mr-2 h-4 w-4" />
                  ) : (
                    <File className="mr-2 h-4 w-4" />
                  )}
                  {item.name}
                </Link>
              </TableCell>
              <TableCell>
                <div className="flex space-x-2 justify-end">
                  {(item.downloadUrl || item.canDeleteDataset) && (
                    <DropdownMenu>
                      {item.previewUrl && (
                        <TooltipProvider>
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <a
                                href={item.previewUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex"
                              >
                                <Button
                                  variant={"outline"}
                                  className="cursor-pointer"
                                  size="sm"
                                >
                                  <Eye className="mr-2 h-4 w-4" />
                                  Preview
                                </Button>
                              </a>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>
                                Please use HVC1 or AVC1 codec for videos to be
                                visible in LeRobot dataset viewer.
                              </p>
                            </TooltipContent>
                          </Tooltip>
                        </TooltipProvider>
                      )}

                      <DropdownMenuTrigger asChild className="cursor-pointer">
                        <Button variant="outline" size="sm">
                          <MoreVertical className="h-4 w-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        {item.huggingfaceUrl && (
                          <DropdownMenuItem asChild>
                            <a
                              href={item.huggingfaceUrl}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="flex items-center cursor-pointer"
                            >
                              <ExternalLink className="mr-2 h-4 w-4" />
                              See on Hugging Face Hub
                            </a>
                          </DropdownMenuItem>
                        )}
                        {item.downloadUrl && item.is_dataset_dir && (
                          <DropdownMenuItem
                            onClick={() => handlePushToHub(item)}
                            className="cursor-pointer"
                          >
                            <ArrowUpFromLine className="mr-2 h-4 w-4" />
                            Push to Hugging Face Hub
                          </DropdownMenuItem>
                        )}
                        {item.downloadUrl && (
                          <DropdownMenuItem asChild>
                            <a
                              href={item.downloadUrl}
                              className="flex items-center cursor-pointer"
                            >
                              <Download className="mr-2 h-4 w-4" />
                              Download
                            </a>
                          </DropdownMenuItem>
                        )}
                        {item.canDeleteDataset && (
                          <>
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              onClick={() => handleDeleteDataset(item)}
                              className="text-red-500 focus:text-destructive cursor-pointer"
                            >
                              <Trash2 className="mr-2 h-4 w-4 text-red-500" />
                              Delete dataset
                            </DropdownMenuItem>
                          </>
                        )}
                      </DropdownMenuContent>
                    </DropdownMenu>
                  )}
                </div>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>

      {data.episode_ids && data.episode_ids.length > 0 && (
        <div className="mt-6">
          <div className="flex items-end gap-2">
            <Select
              value={selectedEpisode}
              name="episode_id"
              onValueChange={setSelectedEpisode}
            >
              <SelectTrigger id="episode-select">
                <SelectValue placeholder="Select Episode ID" />
              </SelectTrigger>
              <SelectContent>
                {data.episode_ids.map((episode) => (
                  <SelectItem key={episode} value={episode.toString()}>
                    Episode {episode}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              type="button"
              variant="destructive"
              onClick={() => {
                if (!selectedEpisode) return;
                setConfirmEpisodeDeleteOpen(true);
              }}
            >
              <Trash2 className="mr-2 h-4 w-3" />
              Delete Episode
            </Button>
            {isRobotConnected && (
              <Button
                type="button"
                variant="default"
                onClick={async () => {
                  await handleReplayEpisode();
                }}
              >
                <Repeat className="mr-2 h-4 w-3" />
                Replay Episode
              </Button>
            )}
          </div>
        </div>
      )}

      {/* Dataset deletion dialog */}
      <Dialog open={confirmDeleteOpen} onOpenChange={setConfirmDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Delete Dataset</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete the dataset{" "}
              <strong>{selectedItem?.name}</strong>? This action cannot be
              undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmDeleteOpen(false)}
            >
              Cancel
            </Button>
            <Button variant="destructive" onClick={handleConfirmDelete}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Episode deletion dialog */}
      <Dialog
        open={confirmEpisodeDeleteOpen}
        onOpenChange={setConfirmEpisodeDeleteOpen}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Delete Episode</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete episode{" "}
              <strong>{selectedEpisode}</strong>? This action cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setConfirmEpisodeDeleteOpen(false)}
              disabled={loadingDeleteEpisode}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={async () => {
                setLoadingDeleteEpisode(true);
                await handleDeleteEpisode();
                setConfirmEpisodeDeleteOpen(false);
                setLoadingDeleteEpisode(false);
                redirect(path);
              }}
              disabled={loadingDeleteEpisode}
            >
              {loadingDeleteEpisode ? (
                <LoaderCircle> Deleting...</LoaderCircle>
              ) : (
                "Delete"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
