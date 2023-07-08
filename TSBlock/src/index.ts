import { mathAdd } from "myPackage";
import { mathMinus } from "./customCode";

class Block {
    constructor(private data: string) {}
    static hello() {
        console.log("hi");
        mathAdd(3,4); mathMinus(3,4);
        return 'hi';
    }
}