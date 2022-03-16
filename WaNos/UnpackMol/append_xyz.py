with open("structures.xyz", "w") as outfile:
  for filename in list:
    with open(filename) as infile:
      contents = infile.read()
      outfile.write(contents)
      outfile.write("\n")
