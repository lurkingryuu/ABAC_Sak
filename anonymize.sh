#!/bin/bash

# Configuration
SOURCE_DIR="."
DEST_NAME="ABAC_Sak_Anonymous"
TMP_DIR="build_anonymous"
OUTPUT_ZIP="../${DEST_NAME}.zip"

echo "🚀 Starting anonymization process..."

# 1. Clean previous builds if any
rm -rf "${TMP_DIR}"
rm -f "${OUTPUT_ZIP}"

# 2. Create temporary build directory
mkdir -p "${TMP_DIR}/${DEST_NAME}"

# 3. Copy files using rsync while explicitly excluding sensitive and local items
echo "📦 Copying files & stripping sensitive hidden files/directories..."
rsync -a \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='.env' \
    --exclude='usage_tracking.db' \
    --exclude='uv.lock' \
    --exclude='*.zip' \
    --exclude='build_anonymous' \
    --exclude='anonymize.sh' \
    --exclude='.python-version' \
    --exclude='*.DS_Store' \
    "${SOURCE_DIR}/" "${TMP_DIR}/${DEST_NAME}/"

# 4. Deep Scrubbing of code and configuration files
echo "🕵️  Scrubbing personal identifiers, local paths, and affiliations..."
cd "${TMP_DIR}/${DEST_NAME}" || { echo "Failed to enter temp directory"; exit 1; }

# Find all regular text files (ignoring images/binaries) to perform text scrubbing
find . -type f -not -path "*/\.*" -not -name "*.png" -not -name "*.jpg" -not -name "*.json" | while read -r file; do
    if file "$file" | grep -q text; then
        # Replace local paths
        sed -i '' 's|/Users/karthik/work/mtp/temp/saket/ABAC_Sak|/path/to/project/ABAC_Sak|g' "$file"
        sed -i '' 's|/Users/karthik|/home/anonymous_user|g' "$file"
        
        # Replace Usernames/Authors (case-insensitive for 'saket' and 'karthik' where possible, 
        # macOS sed doesn't support \I flag easily, so we use exact case replacements common in code)
        sed -i '' 's/karthik/anonymous_user/g' "$file"
        sed -i '' 's/Karthik/Anonymous_User/g' "$file"
        sed -i '' 's/saket/anonymous_author/g' "$file"
        sed -i '' 's/Saket/Anonymous_Author/g' "$file"
    fi
done

# Remove any License or Signature files to enforce double-blind
echo "📄 Removing LICENSE and authorship files..."
find . -type f -iname "LICENSE*" -exec rm -f {} +
find . -type f -iname "CITATION*" -exec rm -f {} +
find . -type f -iname "AUTHORS*" -exec rm -f {} +

cd ../..

# 5. Create the zip archive
echo "🗜️ Zipping the clean repository..."
cd "${TMP_DIR}" || exit 1
zip -r "archive.zip" "${DEST_NAME}/" > /dev/null
cd ..

# Move the zip out to the parent directory
mv "${TMP_DIR}/archive.zip" "${OUTPUT_ZIP}"

# 6. Clean up
echo "🧹 Cleaning up temporary directory..."
rm -rf "${TMP_DIR}"

echo "✅ Success! Double-blind ready package created at:"
# Get absolute path of the output zip for clear printing
ABS_PATH=$(cd "$(dirname "${OUTPUT_ZIP}")" && pwd)/$(basename "${OUTPUT_ZIP}")
echo "   ${ABS_PATH}"
