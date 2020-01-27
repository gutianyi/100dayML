def mid(x, y, z):
    m = z
    if y < z:
        if x < y:
            m = y       # set a bug here: m = z
        elif x < z:
            m = x       # set a bug here: m = y
    elif x > y:
        m = y           # set a bug here: m = x
    elif x > z:
        m = x           # set a bug here: m = y
    return m
