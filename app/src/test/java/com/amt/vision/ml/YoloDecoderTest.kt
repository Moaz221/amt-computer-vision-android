package com.amt.vision.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class YoloDecoderTest {
    private val labels = listOf("mask", "no mask")

    @Test fun decodesBoxBackToOriginalImage() {
        // 200x100 image into a 480 square: scale 2.4, padY 120. Box centered at (240,240) px, 240x120 wide.
        val row = floatArrayOf(0.5f, 0.5f, 0.5f, 0.25f, 0.9f, 0.1f, 0.8f)
        val d = YoloDecoder.decode(arrayOf(row), labels, 480, 200, 100)
        assertEquals(1, d.size)
        assertEquals("no mask", d[0].label)
        assertEquals(50f, d[0].left, 0.01f)
        assertEquals(25f, d[0].top, 0.01f)
        assertEquals(150f, d[0].right, 0.01f)
        assertEquals(75f, d[0].bottom, 0.01f)
        assertEquals(0.72f, d[0].score, 0.001f)
    }

    @Test fun junkRowWithHugeSizeDoesNotSwitchToPixelUnits() {
        // Normalized export where an unconfident row has a width above 1.5: must stay normalized.
        val good = floatArrayOf(0.5f, 0.5f, 0.4f, 0.6f, 0.9f, 0.9f, 0.1f)
        val junk = floatArrayOf(0.5f, 0.5f, 2.0f, 2.0f, 0.01f, 0.5f, 0.5f)
        val d = YoloDecoder.decode(arrayOf(good, junk), labels, 480, 128, 128)
        assertEquals(1, d.size)
        assertEquals(38.4f, d[0].left, 0.5f)
        assertEquals(89.6f, d[0].right, 0.5f)
        assertEquals(25.6f, d[0].top, 0.5f)
        assertEquals(102.4f, d[0].bottom, 0.5f)
    }

    @Test fun weakMaskInsideStrongerNoMaskFaceIsDropped() {
        val face = floatArrayOf(0.5f, 0.5f, 0.5f, 0.6f, 0.95f, 0.05f, 0.95f)   // no mask, big
        val glasses = floatArrayOf(0.5f, 0.4f, 0.3f, 0.1f, 0.9f, 0.7f, 0.1f)   // mask-ish, inside face
        val d = YoloDecoder.decode(arrayOf(face, glasses), labels, 480, 480, 480)
        assertEquals(1, d.size)
        assertEquals("no mask", d[0].label)
    }

    @Test fun maskNeedsItsOwnHigherScore() {
        val weakMask = floatArrayOf(0.5f, 0.5f, 0.3f, 0.3f, 0.5f, 0.6f, 0.1f) // score 0.30
        val d = YoloDecoder.decode(arrayOf(weakMask), labels, 480, 480, 480, 0.2f, 0.45f, mapOf(0 to 0.45f))
        assertEquals(0, d.size)
    }

    @Test fun dropsLowConfidenceAndSuppressesOverlaps() {
        val a = floatArrayOf(0.5f, 0.5f, 0.4f, 0.4f, 0.9f, 0.9f, 0.1f)
        val b = floatArrayOf(0.51f, 0.5f, 0.4f, 0.4f, 0.8f, 0.9f, 0.1f) // same object, lower score
        val c = floatArrayOf(0.2f, 0.2f, 0.1f, 0.1f, 0.1f, 0.9f, 0.1f)  // below threshold
        val d = YoloDecoder.decode(arrayOf(a, b, c), labels, 480, 480, 480)
        assertEquals(1, d.size)
        assertTrue(d[0].score > 0.8f)
    }

    @Test fun differentClassesAreNotSuppressedAgainstEachOther() {
        val a = floatArrayOf(0.5f, 0.5f, 0.4f, 0.4f, 0.9f, 0.9f, 0.1f)
        val b = floatArrayOf(0.15f, 0.15f, 0.1f, 0.1f, 0.9f, 0.1f, 0.9f)
        assertEquals(2, YoloDecoder.decode(arrayOf(a, b), labels, 480, 480, 480).size)
    }
}
