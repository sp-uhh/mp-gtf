### This Python script generates the Multi-Phase Gammatone Filterbank as described in [1]
#    for usage with Conv-TasNet
#
#    [1] D. Ditter and T. Gerkmann, “A Multi-Phase Gammatone Filterbank for Speech 
#        Separation via TasNet”, arXiv preprint arXiv:1910.11615, Oct. 2019. 
#        Available: https://arxiv.org/abs/1910.11615

import numpy as np
import matplotlib.pyplot as plt

### UTILITY FUNCTIONS ###

def erbScale2freqHz(fErb):
    # Convert frequency on ERB scale to frequency in Hertz
    fHz = (np.exp(fErb/9.265)-1)*24.7*9.265
    return fHz

def freqHz2erbScale(fHz):
    # Convert frequency in Hertz to frequency on ERB scale
    fErb = 9.265*np.log(1+fHz/(24.7*9.265))
    return fErb

def normalizeFilters(filterbank):
    # Normalizes a filterbank such that all filters
    # have the same root mean square (RMS).
    rmsPerFilter = np.sqrt(np.mean(np.square(filterbank), axis=1))
    rmsNormalizationValues = 1. / (rmsPerFilter/np.amax(rmsPerFilter))
    normalizedFilterbank = filterbank * rmsNormalizationValues[:, np.newaxis]
    return normalizedFilterbank

### GAMMATONE IMPULSE RESPONSE ###

def gammatoneImpulseResponse(samplerateHz, lengthInSeconds, centerFreqHz, phaseShift):
    # Generate single parametrized gammatone filter
    p = 2 # filter order
    erb = 24.7 + 0.108*centerFreqHz # equivalent rectangular bandwidth
    divisor = (np.pi * np.math.factorial(2*p-2) * np.power(2, float(-(2*p-2))) )/ np.square(np.math.factorial(p-1))
    b = erb/divisor # bandwidth parameter
    a = 1.0 # amplitude. This is varied later by the normalization process.
    L = int(np.floor(samplerateHz*lengthInSeconds))
    t = np.linspace(1./samplerateHz, lengthInSeconds, L)
    gammatoneIR = a * np.power(t, p-1)*np.exp(-2*np.pi*b*t) * np.cos(2*np.pi*centerFreqHz*t + phaseShift)
    return gammatoneIR

### MP-GTF CONSTRUCTION ###

def generateMPGTF(samplerateHz, lengthInSeconds, N):
    # Set parameters
    centerFreqHzMin = 100
    nCenterFreqs = 24
    L = int(np.floor(samplerateHz*lengthInSeconds))

    # Initialize variables
    index = 0
    filterbank = np.zeros((N, L))
    currentCenterFreqHz = centerFreqHzMin

    # Determine number of phase shifts per center frequency
    phasePairCount = (np.ones(nCenterFreqs)*np.floor(N/2/nCenterFreqs)).astype(int)
    remainingPhasePairs = ((N-np.sum(phasePairCount)*2)/2).astype(int)
    if remainingPhasePairs > 0:
        phasePairCount[:remainingPhasePairs] = phasePairCount[:remainingPhasePairs]+1

    # Generate all filters for each center frequencies
    for i in range(nCenterFreqs):
        # Generate all filters for all phase shifts
        for phaseIndex in range(phasePairCount[i]):
            # First half of filtes: Phaseshifts in [0,pi)
            currentPhaseShift = np.float(phaseIndex) / phasePairCount[i] * np.pi
            filterbank[index, :] = gammatoneImpulseResponse(samplerateHz, lengthInSeconds, currentCenterFreqHz, currentPhaseShift)
            index = index+1

        # Second half of filtes: Phaseshifts in [pi, 2*pi)
        filterbank[index:index+phasePairCount[i], :] = -filterbank[index-phasePairCount[i]:index, :]

        # Prepare for next center frequency
        index = index + phasePairCount[i]
        currentCenterFreqHz = erbScale2freqHz(freqHz2erbScale(currentCenterFreqHz)+1)

    filterbank = normalizeFilters(filterbank)
    return filterbank

### GENERATE AND PLOT ###

def generateExampleAndPlot():
    # Set parameters
    N = 128 # Number of filters
    samplerateInHz = 8000
    filterLengthInSeconds = 0.002

    # Generate the MP-GTF
    filterbank = generateMPGTF(samplerateInHz, filterLengthInSeconds, N)

    # Plot MP-GTF
    plt.figure()
    plt.imshow(filterbank, origin='bottom', cmap='bwr')
    plt.xlabel('Time (in samples)')
    plt.ylabel('Filter index n')

if __name__ == '__main__':
    generateExampleAndPlot()
