#!/bin/bash

# Function to rename a file or directory
rename_item() {
    local old_name="$1"
    local dir_path=$(dirname "$old_name")
    local base_name=$(basename "$old_name")

    # Skip if it's a hidden file or directory
    if [[ $base_name == .* ]]; then
        return
    fi

    # Check if it's a directory
    if [ -d "$old_name" ]; then
        # For directories, replace all non-alphanumeric characters with underscores
        local new_name_part=$(echo "$base_name" | sed -e 's/[^[:alnum:]]/_/g' -e 's/__*/_/g' -e 's/^_//g' -e 's/_$//g')
        local new_name="$dir_path/$new_name_part"
    else
        # For files, split the base name into name and extension
        local name_part="${base_name%.*}"
        local extension="${base_name##*.}"

        # If there's no extension, set extension to empty
        if [ "$name_part" = "$extension" ]; then
            extension=""
        else
            extension=".$extension"
        fi

        # Process the name part
        local new_name_part=$(echo "$name_part" | sed -e 's/[^[:alnum:]]/_/g' -e 's/__*/_/g' -e 's/^_//g' -e 's/_$//g')

        # Combine the processed name part with the extension
        local new_name="$dir_path/${new_name_part}${extension}"
    fi
    
    if [ "$old_name" != "$new_name" ]; then
        if [ "$DRY_RUN" = true ]; then
            echo "Would rename: $old_name -> $new_name"
        else
            mv -n "$old_name" "$new_name"
            echo "Renamed: $old_name -> $new_name"
        fi
    fi
}

# Recursively rename files and directories
rename_recursive() {
    local dir="$1"
    
    # Use find with -P to avoid following symlinks
    # Rename directories first (in reverse depth order)
    find -P "$dir" -depth -type d ! -name ".*" | while read -r directory; do
        rename_item "$directory"
    done
    
    # Then rename files
    find -P "$dir" -type f ! -name ".*" | while read -r file; do
        rename_item "$file"
    done
}

# Print usage information
print_usage() {
    echo "Usage: $0 [-d] [-h] [directory]"
    echo "  -d    Dry run (print what would be done without actually renaming)"
    echo "  -h    Display this help message"
    echo "  directory    The directory to process (default: current directory)"
}

# Main script
DRY_RUN=false
TARGET_DIR="."

# Parse command-line options
while getopts ":dh" opt; do
    case ${opt} in
        d )
            DRY_RUN=true
            ;;
        h )
            print_usage
            exit 0
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            print_usage
            exit 1
            ;;
    esac
done
shift $((OPTIND -1))

# Check if a directory was specified
if [ $# -eq 1 ]; then
    TARGET_DIR="$1"
fi

# Verify the target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist." 1>&2
    exit 1
fi

# Start the renaming process
if [ "$DRY_RUN" = true ]; then
    echo "Performing dry run..."
fi

rename_recursive "$TARGET_DIR"

if [ "$DRY_RUN" = true ]; then
    echo "Dry run complete. No files were actually renamed."
else
    echo "Renaming complete!"
fi
