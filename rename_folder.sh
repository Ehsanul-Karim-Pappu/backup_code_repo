#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to show colored diff
show_diff() {
    local old_name="$1"
    local new_name="$2"
    local old_output=""
    local new_output=""
    local i=0
    local j=0

    while [ $i -lt ${#old_name} ] || [ $j -lt ${#new_name} ]; do
        if [ $i -ge ${#old_name} ]; then
            # Remaining characters in new_name are additions
            new_output+="${GREEN}${new_name:$j}${NC}"
            break
        elif [ $j -ge ${#new_name} ]; then
            # Remaining characters in old_name are deletions
            old_output+="${RED}${old_name:$i}${NC}"
            break
        elif [ "${old_name:$i:1}" = "${new_name:$j:1}" ]; then
            # Characters are the same
            old_output+="${old_name:$i:1}"
            new_output+="${new_name:$j:1}"
            i=$((i+1))
            j=$((j+1))
        else
            # Characters differ
            if [[ "${old_name:$i:1}" =~ [^[:alnum:]] && "${new_name:$j:1}" = "_" ]]; then
                # Non-alphanumeric replaced with underscore
                old_output+="${YELLOW}${old_name:$i:1}${NC}"
                new_output+="${GREEN}_${NC}"
                i=$((i+1))
                j=$((j+1))
            elif [[ "${old_name:$i:1}" =~ [^[:alnum:]] ]]; then
                # Non-alphanumeric character removed
                old_output+="${RED}${old_name:$i:1}${NC}"
                i=$((i+1))
            else
                # New character added
                new_output+="${GREEN}${new_name:$j:1}${NC}"
                j=$((j+1))
            fi
        fi
    done

    echo -e "$old_output ->"
    echo -e "$new_output"
}

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
            echo -e "${BLUE}Would rename:${NC}"
            show_diff "$(basename "$old_name")" "$(basename "$new_name")"
            echo
        else
            mv -n "$old_name" "$new_name"
            echo -e "${GREEN}Renamed:${NC} $(basename "$old_name") -> $(basename "$new_name")"
        fi
    fi
}

# The rest of the script remains unchanged
# ...

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
    echo -e "${BLUE}Performing dry run...${NC}"
fi

rename_recursive "$TARGET_DIR"

if [ "$DRY_RUN" = true ]; then
    echo -e "${BLUE}Dry run complete. No files were actually renamed.${NC}"
else
    echo -e "${GREEN}Renaming complete!${NC}"
fi
