def mid(x, y, z):
    m = z
    if y < z:
        if x < y:
            m = y
        elif x < z:
            m = y  # ***BUG***
    elif x > y:
        m = y
    elif x > z:
        m = x
    return m
