# Download Data



all: season1.zip
	unzip season1.zip


season1.zip:
	wget https://github.com/tudelft/MonoRaceGate/releases/download/v1.0.0/season1.zip


zip:
	zip -r season1.zip . -i '*.jpg' '*.png'
	find . -type f -name "*.png" -delete
	find . -type f -name "*.jpg" -delete
