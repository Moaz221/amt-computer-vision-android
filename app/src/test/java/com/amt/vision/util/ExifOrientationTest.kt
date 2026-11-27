package com.amt.vision.util

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ExifOrientationTest {
    @Test fun normalAndUnknownDoNothing() {
        assertTrue(ExifTransform.fromOrientation(1).isIdentity)
        assertTrue(ExifTransform.fromOrientation(0).isIdentity)
        assertTrue(ExifTransform.fromOrientation(99).isIdentity)
    }

    @Test fun uprightPhoneShotRotates90() {
        val t = ExifTransform.fromOrientation(6)
        assertEquals(90, t.degrees); assertEquals(false, t.mirror)
    }

    @Test fun rotationSwapsSizeOnlyFor90And270() {
        assertEquals(300 to 400, ExifTransform.uprightSize(400, 300, ExifTransform.fromOrientation(6)))
        assertEquals(300 to 400, ExifTransform.uprightSize(400, 300, ExifTransform.fromOrientation(8)))
        assertEquals(400 to 300, ExifTransform.uprightSize(400, 300, ExifTransform.fromOrientation(3)))
    }

    @Test fun mirroredVariants() {
        assertEquals(ExifTransform(0, true), ExifTransform.fromOrientation(2))
        assertEquals(ExifTransform(180, true), ExifTransform.fromOrientation(4))
        assertEquals(ExifTransform(90, true), ExifTransform.fromOrientation(5))
        assertEquals(ExifTransform(270, true), ExifTransform.fromOrientation(7))
    }
}
