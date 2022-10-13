




/**
 * Java Arrays
 * (▰˘◡˘▰)
 * This video talks about Java Array!
 *
 * I am going to create an array of cats 🐱
 * 1) Give each cat a name
 * 2) Each cat will "Meoooow~"
 *
 * A lot of explanation in this video is from a book titled
 *      "Head First Java" by Kathy Sierra & Bert Bates
 *
 * Please 🌟PAUSE🌟 the video any time you want to study the code written.
 */















public class catArray {

    String name;  // where cats' names will be stored

    public void meow() {
        /**
         * method describing how each cat will meow after each cat's name
         */
        System.out.println(name + " says Meoooow~");
    }


    public static void main (String[] args) {
        /**
         * MAIN method to create an array of 3 cats AND
         * each cat will meow
         */

        // creating a cat array with length 3
        catArray[] cat = new catArray[3];

        // create each cat
        cat[0] = new catArray();
        cat[1] = new catArray();
        cat[2] = new catArray();

        // give each cat a name
        cat[0].name = "Tom";
        cat[1].name = "Jim";
        cat[2].name = "Loki";

        // WHILE loop to make each cat meow in turn
        int i = 0;  // initialize i from 0
        while (i < cat.length) { // cat.length is 3 since we have 3 cats
            cat[i].meow();
            i = i + 1;
        }
    }
}











/**
 * Compile step:
 * javac catArray.java
 *      notice a new .class file appeared under V114
 *
 * Run
 * java catArray
 *      each cat meowing~








 * This is the end of "Java Arrays" video~
 *
 * Hope you enjoyed it!
 * Thank you for watching ◎[▪‿▪]◎
 */









