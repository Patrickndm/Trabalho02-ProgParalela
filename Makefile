CC     = gcc
MPICC  = mpicc
CFLAGS = -O2 -Wall -Wextra

all: blur_seq blur_par

blur_seq: sequencial.c
	$(CC) $(CFLAGS) -o $@ $<

blur_par: paralelo.c
	$(MPICC) $(CFLAGS) -o $@ $<

clean:
	rm -f blur_seq blur_par

.PHONY: all clean
