
# Google Drive file ID
file_id="1c_issruQovURZg7je-r8QVur3Viubnjm"

# Output zip file name
output_zip="data.zip"

# Download the zip file using gdown
gdown "https://drive.google.com/uc?id=$file_id" -O $output_zip

echo "Download complete: $output_zip"
