def mid(x, y, z):
    m = z
    if y < z:
        if x < y:
            m = z       # 1st bug: here should be m = y
        elif x < z:
            m = x
    elif x > y:
        m = y
    elif x > z:
        m = x
    return m
