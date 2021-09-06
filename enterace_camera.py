import requests


if __name__ == "__main__":
    with open('fake_users/Bradley_Cooper5.jpeg', 'rb') as pic:
        r = requests.post('http://localhost:5000/enterance',
                          files={'file': pic})
        print(r.text)
