



/**
 * Intro to Java
 * (▰˘◡˘▰)
 * This video introduces Java, just to help us get started~!
 *      < This is my 1st video on Java >
 *
 * I am creating a class called "Dog" which describes how the dog behaves
 * Then, I am creating another class with MAIN method,
 * which will actually create our dog! 🐶
 *
 * A lot of explanation in this video is from a book titled
 *      "Head First Java" by Kathy Sierra & Bert Bates
 *
 *
 * Please 🌟PAUSE🌟 the video any time you want to study the code written.
 */


















public class Dog {
    /**
     * This class describes how our dog behaves!
     */
    String name;     //  variable to store our dog's name

    void bark() {    // method showing how the dog barks
        System.out.println("Ruff Ruff");
    }

    void selfIntro() { // method telling us the dog's name, using name from above
        System.out.println("My name is " + name);
    }
}











class DogTestDrive {
    /**
     * class which actually creates our dog! 🐶
     */
    public static void main(String[] args) {
        /**
         * Main method to create our dog!
         */

        Dog d = new Dog();  // create our dog

        d.name = "Snoopy";  // our dog's name is Snoopy

        d.bark();           // let our dog bark
        d.selfIntro();      // tell us the dog's name
    }
}














/**
Let's Compile and Run our Java code!

Compile
javac Dog.java        <-- Execute this command in Terminal
                          Notice 2 new files with .class extension under 'src'

Run
java DogTestDrive     <-- Execute this command in Terminal

Notice our dog is barking AND
our dog's name is Snoopy!
 */













/**
 * This is the end of "Intro to Java" video~
 *
 * Hope you enjoyed it!
 * Thank you for watching ◎[▪‿▪]◎
 */





