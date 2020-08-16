import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    args = parser.parse_args()

    with open(args.text, "r") as f:
        for line in f.readlines():
            print("{ID}{data}{endpoint}".format(ID=line.split(" ")[0]+" {", data="} {".join(line.replace("\n","").split(" ")[1:-1]), endpoint="}"))


if __name__ == "__main__":
    main()