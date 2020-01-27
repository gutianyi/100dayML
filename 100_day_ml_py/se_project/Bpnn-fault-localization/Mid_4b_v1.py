def mid(x, y, z):
    m = z
    if y < z:
        if x < y:
            m = z       # 1st bug: here should be m = y
        elif x < z:
            m = y       # 2nd bug: here should be m = x
    elif x > y:
        m = x           # 3rd bug: here should be m = y
    elif x > z:
        m = y           # 4th bug: here should be m = x
    return m
