package com.amt.vision.ml

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Checks the app's preprocessing against the reference implementation in
 * common/preprocess.py. The expected numbers below are that file's output for
 * the same four pixels, so a change here means the phone and the trainer have
 * drifted apart again.
 */
class PreprocessingTest {

    // Packed ARGB, exactly what Bitmap.getPixels hands back.
    private val white = 0xFFFFFFFF.toInt()
    private val black = 0xFF000000.toInt()
    private val red = 0xFFFF0000.toInt()
    private val teal = 0xFF0A141E.toInt() // r=10 g=20 b=30

    private val pixels = intArrayOf(white, black, red, teal)

    private fun run(mode: String): Array<Array<FloatArray>> =
        Preprocessing.toInput(mode, pixels, width = 2, height = 2)[0]

    private fun assertPixel(expected: FloatArray, actual: FloatArray) {
        assertEquals(3, actual.size)
        for (i in expected.indices) {
            assertEquals("channel $i", expected[i], actual[i], 1e-4f)
        }
    }

    @Test
    fun rawKeepsRgb0to255() {
        val out = run("raw")
        assertPixel(floatArrayOf(255f, 255f, 255f), out[0][0])
        assertPixel(floatArrayOf(0f, 0f, 0f), out[0][1])
        assertPixel(floatArrayOf(255f, 0f, 0f), out[1][0])
        assertPixel(floatArrayOf(10f, 20f, 30f), out[1][1])
    }

    @Test
    fun rescaleDividesBy255() {
        val out = run("rescale")
        assertPixel(floatArrayOf(1f, 1f, 1f), out[0][0])
        assertPixel(floatArrayOf(0f, 0f, 0f), out[0][1])
        assertPixel(floatArrayOf(1f, 0f, 0f), out[1][0])
        assertPixel(floatArrayOf(0.03921569f, 0.07843138f, 0.1176471f), out[1][1])
    }

    @Test
    fun tfMapsToMinusOneToOne() {
        val out = run("tf")
        assertPixel(floatArrayOf(1f, 1f, 1f), out[0][0])
        assertPixel(floatArrayOf(-1f, -1f, -1f), out[0][1])
        assertPixel(floatArrayOf(1f, -1f, -1f), out[1][0])
        assertPixel(floatArrayOf(-0.9215686f, -0.8431373f, -0.7647059f), out[1][1])
    }

    @Test
    fun effnetCentersOn127() {
        val out = run("effnet")
        assertPixel(floatArrayOf(1f, 1f, 1f).map { 128f / 128f }.toFloatArray(), out[0][0])
        assertPixel(floatArrayOf(-127f / 128f, -127f / 128f, -127f / 128f), out[0][1])
        assertPixel(floatArrayOf(-117f / 128f, -107f / 128f, -97f / 128f), out[1][1])
    }

    @Test
    fun caffeSwapsToBgrAndSubtractsTheMean() {
        val out = run("caffe")
        assertPixel(floatArrayOf(151.061f, 138.221f, 131.32f), out[0][0])
        assertPixel(floatArrayOf(-103.939f, -116.779f, -123.68f), out[0][1])
        // pure red: blue channel first, and it is the high one
        assertPixel(floatArrayOf(-103.939f, -116.779f, 131.32f), out[1][0])
        assertPixel(floatArrayOf(-73.939f, -96.779f, -113.68f), out[1][1])
    }

    @Test
    fun modeAliasesResolve() {
        assertEquals("rescale", Preprocessing.normalizeMode(null))
        assertEquals("raw", Preprocessing.normalizeMode("none"))
        assertEquals("rescale", Preprocessing.normalizeMode("div255"))
        assertEquals("tf", Preprocessing.normalizeMode("-1..1"))
        assertEquals("caffe", Preprocessing.normalizeMode("BGR"))
        assertEquals("rescale", Preprocessing.normalizeMode(""))
    }

    @Test
    fun unknownModeIsRejectedInsteadOfGuessed() {
        assertThrows(IllegalArgumentException::class.java) {
            Preprocessing.normalizeMode("imagenet")
        }
    }

    @Test
    fun outputModesResolve() {
        assertEquals("softmax", Postprocessing.normalizeOutput(null))
        assertEquals("softmax", Postprocessing.normalizeOutput("probabilities"))
        assertEquals("logits", Postprocessing.normalizeOutput("scores"))
        assertThrows(IllegalArgumentException::class.java) {
            Postprocessing.normalizeOutput("sigmoid")
        }
    }

    @Test
    fun softmaxOfProbabilitiesFlattensThem() {
        // Why a softmax output must not be softmaxed again: the ranking
        // survives but the scores collapse toward 1/N.
        val probabilities = floatArrayOf(0.90f, 0.06f, 0.04f)
        val again = Postprocessing.softmax(probabilities)
        assertTrue(again.max() < probabilities.max())
        assertEquals(1f, again.sum(), 1e-5f)
        assertEquals(0, again.indices.maxByOrNull { again[it] }!!)
    }

    @Test
    fun softmaxOfLogitsIsAProbabilityDistribution() {
        val logits = floatArrayOf(2.5f, -1f, 0.3f)
        val probs = Postprocessing.softmax(logits)
        assertEquals(1f, probs.sum(), 1e-5f)
        assertEquals(0, probs.indices.maxByOrNull { probs[it] }!!)
        assertArrayEquals(floatArrayOf(0.8764f, 0.0265f, 0.0971f), probs, 0.001f)
    }
}
