form Test command line calls
    sentence file ''
endform

Read from file... 'file$'
method$ = "Spectral subtraction"
smoothing = 20
filtered = Remove noise: 0.0, 0.0, 0.025, 80, 10000, smoothing, method$
Save as WAV file... 'file$'