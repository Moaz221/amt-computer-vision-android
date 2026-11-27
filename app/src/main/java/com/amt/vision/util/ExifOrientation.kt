package com.amt.vision.util

/**
 * EXIF orientation (tag values 1..8) as "rotate clockwise by [degrees], then mirror left-right if
 * [mirror]". Pure Kotlin so it is unit tested on the JVM.
 */
data class ExifTransform(val degrees: Int, val mirror: Boolean) {
    val isIdentity: Boolean get() = degrees == 0 && !mirror

    companion object {
        fun fromOrientation(orientation: Int): ExifTransform = when (orientation) {
            2 -> ExifTransform(0, true)      // mirror
            3 -> ExifTransform(180, false)
            4 -> ExifTransform(180, true)    // flip vertical
            5 -> ExifTransform(90, true)     // transpose
            6 -> ExifTransform(90, false)    // phone held upright (most common)
            7 -> ExifTransform(270, true)    // transverse
            8 -> ExifTransform(270, false)
            else -> ExifTransform(0, false)
        }

        /** Size of the upright image for a stored width/height. */
        fun uprightSize(w: Int, h: Int, t: ExifTransform): Pair<Int, Int> =
            if (t.degrees == 90 || t.degrees == 270) h to w else w to h
    }
}
