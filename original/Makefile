
CFLAGS=-O3 -fopenmp -I.
LIBS=-lm

extinguishing: extinguishing.c
	gcc $(CFLAGS) $< -o $@ $(LIBS)

debug: extinguishing.c
	gcc-8  $(CFLAGS) -DDEBUG $< -o $@ $(LIBS)

time: extinguishing.c
	gcc-8  $(CFLAGS) -DTIME $< -o $@ $(LIBS)

clean:
	rm -f extinguishing debug
