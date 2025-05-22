form Transform mp3 to textgrid
	sentence Directory ./
	sentence Filename ./
	sentence Type ./
	real Threshold -15.0
	real Minsil 0.2
	real Minsound 0.05
endform

Read from file... 'directory$''filename$''type$'
soundname$ = selected$ ("Sound")
To Intensity... 60 0 
To TextGrid (silences)... threshold minsil minsound silent sounding
Save as text file: directory$ + "" + filename$ + ".TextGrid"