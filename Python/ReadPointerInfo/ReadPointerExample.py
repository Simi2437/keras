import ctypes  # C Types laesst uns quasi unbegrenzt C++ nutzen. https://docs.python.org/3/library/ctypes.html

# Dieses Object soll es ermoeglichen Byteweise Daten aus dem Arbeitsspeicher and der Stelle auszulesen, die durch ein Programm uebergeben wurde.
#
# Stichworte Interoperabilitaet: Wie kann unser Python Skript Daten aus einer anderen Programmiersprache verwenden. https://openbook.rheinwerk-verlag.de/python/37_001.html

# Ein C Pointer besteht im Prinzip aus einem c_int() was einfach die Adresse darstellt.
# i = ctypes.c_int()
# Dieser wird in eine Pointer Instance gegeben.
# Pointer = ctypes.pointer(i)

