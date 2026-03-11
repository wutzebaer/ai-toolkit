#!/bin/bash

# Script zum Resizen aller Bilder auf 720x1280px mit Fill-Modus (ausfüllen statt Ränder)
# Verwendet ImageMagick
# Parallelisiert für schnellere Verarbeitung

# Zielgröße
WIDTH=720
HEIGHT=1280

# Anzahl paralleler Prozesse (Standard: Anzahl CPU-Kerne)
MAX_JOBS=${MAX_JOBS:-$(nproc)}

# Unterstützte Bildformate
# FORMATS="jpg jpeg png webp mp4"
FORMATS="jpg jpeg png"


# Prüfe ob ImageMagick installiert ist
if ! command -v convert &> /dev/null && ! command -v magick &> /dev/null; then
    echo "Fehler: ImageMagick ist nicht installiert."
    echo "Installiere es mit: sudo apt-get install imagemagick"
    exit 1
fi


# Verwende 'magick' falls verfügbar, sonst 'convert'
if command -v magick &> /dev/null; then
    CONVERT_CMD="magick convert"
else
    CONVERT_CMD="convert"
fi

# Eingabe- und Ausgabeverzeichnisse
INPUT_DIR="p1"
OUTPUT_DIR="p2_resized"
OUTPUT_DIR_QUAD="p2_resized_quad"

# Prüfe ob Eingabeverzeichnis existiert
if [ ! -d "$INPUT_DIR" ]; then
    echo "Fehler: Verzeichnis '$INPUT_DIR' existiert nicht."
    exit 1
fi

# Erstelle Ausgabeverzeichnisse falls sie nicht existieren
mkdir -p "$OUTPUT_DIR"
rm -f "$OUTPUT_DIR"/*
mkdir -p "$OUTPUT_DIR_QUAD"
rm -f "$OUTPUT_DIR_QUAD"/*

# Zähler für verarbeitete Bilder (thread-safe mit Datei)
COUNT_FILE=$(mktemp)
echo 0 > "$COUNT_FILE"

# Thread-safe Counter-Increment Funktion
increment_count() {
    flock -x "$COUNT_FILE" sh -c "echo \$((\$(cat $COUNT_FILE) + 1))" > "$COUNT_FILE"
}

# Funktion zum Verarbeiten eines einzelnen Bildes
process_image() {
    local file="$1"
    local filename=$(basename "$file")
    
    echo "Verarbeite: $file"
    
    # Ausgabedateien
    local output_file="$OUTPUT_DIR/$filename"
    local output_file_quad="$OUTPUT_DIR_QUAD/$filename"
    
    # Ermittle Bildabmessungen
    local dimensions=$($CONVERT_CMD "$file" -format "%wx%h" info:)
    local img_width=$(echo $dimensions | cut -d'x' -f1)
    local img_height=$(echo $dimensions | cut -d'x' -f2)
    
    # Bestimme Orientierung: querformat (landscape) oder hochformat (portrait)
    local target_width target_height
    if [ "$img_width" -gt "$img_height" ]; then
        # Querformat: verwende HEIGHTxWIDTH (1280x720)
        target_width=$HEIGHT
        target_height=$WIDTH
    else
        # Hochformat oder quadratisch: verwende WIDTHxHEIGHT (720x1280)
        target_width=$WIDTH
        target_height=$HEIGHT
    fi
    
    # Resize mit Fill-Modus: ^ bedeutet "fill" (ausfüllen), dann crop mit extent
    if $CONVERT_CMD "$file" -resize "${target_width}x${target_height}^" -gravity north -extent "${target_width}x${target_height}" -delay 150 "$output_file" 2>/dev/null; then
        echo "  ✓ Erfolgreich resized: $output_file"
        
        # Erstelle zusätzlich quadratische Version (WIDTHxWIDTH)
        if $CONVERT_CMD "$file" -resize "${WIDTH}x${WIDTH}^" -gravity north -extent "${WIDTH}x${WIDTH}" -delay 150 "$output_file_quad" 2>/dev/null; then
            echo "  ✓ Erfolgreich quadratisch resized: $output_file_quad"
            # Thread-safe Counter-Increment
            increment_count
        else
            echo "  ✗ Fehler beim quadratischen Resizen: $file"
            rm -f "$output_file_quad"
        fi
    else
        echo "  ✗ Fehler beim Resizen: $file"
        rm -f "$output_file"
    fi
}

# Sammle alle zu verarbeitenden Dateien
FILES=()
for ext in $FORMATS; do
    for file in "$INPUT_DIR"/*.$ext "$INPUT_DIR"/*."${ext^^}"; do
        if [ -f "$file" ]; then
            FILES+=("$file")
        fi
    done
done

echo "Gefunden: ${#FILES[@]} Dateien"
echo "Parallele Prozesse: $MAX_JOBS"
echo ""

# Verarbeite Dateien parallel
for file in "${FILES[@]}"; do
    # Warte, bis ein Slot frei ist
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep 0.1
    done
    
    # Starte Verarbeitung im Hintergrund
    process_image "$file" &
done

# Warte auf alle Background-Jobs
wait

# Lese finalen Counter
count=$(cat "$COUNT_FILE")
rm -f "$COUNT_FILE"

echo ""
echo "Fertig! $count Bilder wurden verarbeitet."
