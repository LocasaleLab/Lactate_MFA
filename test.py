import emoji
import numpy as np
import matplotlib.pyplot as plt
import re


def emoji_test():
    emoji_count = 0
    line = "hello 👩🏾‍🎓 emoji hello 👨‍👩‍👦‍👦 how are 😊 you today🙅🏽🙅🏽"
    print(line.encode())
    for word in line:
        if word in emoji.UNICODE_EMOJI:
            emoji_count += 1

    print(emoji_count)


def surrounding_circle():
    def make_circle(p_xloc, q_xloc, s_yloc):
        if p_xloc > q_xloc:
            swap = q_xloc
            q_xloc = p_xloc
            p_xloc = swap
        pq_dist = q_xloc - p_xloc
        an = np.linspace(0, 2 * np.pi, 100)
        circle_x_loc = np.cos(an)
        circle_y_loc = np.sin(an)
        p_circle = [circle_x_loc * pq_dist + p_xloc, circle_y_loc * pq_dist]
        q_circle = [circle_x_loc * pq_dist + q_xloc, circle_y_loc * pq_dist]
        s_xloc = np.arctan(s_yloc / pq_dist)

        fig, ax = plt.subplots()
        ax.plot(p_circle[0], p_circle[1])
        ax.plot(q_circle[0], q_circle[1])


def main():
    # emoji_test()
    surrounding_circle()


if __name__ == '__main__':
    main()
