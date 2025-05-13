# Makefile for running simulation and launching server


# Default output filename when running `make build`
OUTPUT_FILENAME ?= main.bin

# Default target
all: prod

# Run the server for prod settings (able to connect to the Meta Quest). If npm is not installed, it will skip the build step.
prod:
	cd ./teleop/dashboard && ((npm i && npm run build) || echo "npm command failed, continuing anyway") 
	cd teleop && uv run teleop run --simulation=headless --no-telemetry

# Run the server for prod settings with the simulation enabled
prod_sim:
	cd ./teleop && uv run teleop run --simulation=gui --no-telemetry

# Run info command for prod settings
prod_info:
	cd ./teleop && uv run teleop info --opencv --servos

# Run localhost server for dev settings
local:
	cd ./teleop && uv run teleop run --simulation=gui --port=8080 --host=127.0.0.1 --no-telemetry

# For running integration tests
test_server:
	cd ./teleop && uv run teleop run --simulation=headless --only-simulation --simulate-cameras --port=8080 --host=127.0.0.1 --no-telemetry &

# For running the built app in test mode
run_bin_test:
	./teleop/dist/main.bin run --simulation=headless --simulate-cameras --port=8080 --host=127.0.0.1 --no-telemetry &

# For running the built app in prod mode
run_bin:
	./teleop/dist/main.bin run --simulation=headless

# To have the information about the software and the hardware from the built app
info_bin:
	./teleop/dist/main.bin info


# Don't use sudo uv run, it will break CICD
build_pyinstaller:
	cd teleop && \
	uv run pyinstaller \
	--onefile \
	--name $(OUTPUT_FILENAME) \
	--add-data "resources:resources" \
	--add-data "../phosphobot-python/phosphobot/urdf:phosphobot/urdf" \
	--add-data "dashboard/dist:dashboard/dist" \
	--hidden-import teleop \
	--collect-all teleop \
	--collect-all phosphobot \
	--clean -c \
	teleop/main.py && \

clean_build:
	cd ./teleop && rm -rf main.build main.dist main.onefile-build $(OUTPUT_FILENAME)

build_frontend:
	cd ./teleop/dashboard && npm i && npm run build

# Clean up
stop:
	echo "Checking for uv process..."
	ps -ef | grep uv || echo "No uv process running."
	echo "Checking for Python processes..."
	ps -ef | grep python || echo "No Python processes found."
	-killall uv || echo "No uv process found."
	-killall python3 || echo "No python3 process found."
	-killall python3.8 || echo "No python3.8 process found."
	echo "Cleanup complete."

# Clean up hard (kill processes on ports 80 and 8080)
stop_hard:
	echo "Killing all processes listening on port 80..."
	-sudo lsof -i :80 -t | xargs -r sudo kill -9
	echo "Killing all processes listening on port 8020..."
	-sudo lsof -i :8020 -t | xargs -r sudo kill -9
	echo "Killing all processes listening on port 8080..."
	-sudo lsof -i :8080 -t | xargs -r sudo kill -9
	echo "Forceful cleanup of all processes on ports 80, 8020 and 8080 complete."


submodule:
	git submodule update --init --recursive


.PHONY: all sim dev prod stop stop_hard dataset_annotate dataset_convert dataset_push robot_watch test_server build clean_build build_pyinstaller run_bin run_bin_test info_bin