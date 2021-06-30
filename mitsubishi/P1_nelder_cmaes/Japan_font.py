from matplotlib import font_manager
for i in font_manager.fontManager.ttflist:
    if ".ttc" in i.fname:
        print(i)
