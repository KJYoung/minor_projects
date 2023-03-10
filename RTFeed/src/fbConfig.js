import { initializeApp } from "firebase/app";
import { createUserWithEmailAndPassword, getAuth, GithubAuthProvider, GoogleAuthProvider, onAuthStateChanged, signInWithEmailAndPassword, signInWithPopup, signOut, updateProfile } from "firebase/auth";
import { addDoc, collection, deleteDoc, doc, getDocs, getFirestore, onSnapshot, orderBy, query, updateDoc, where } from "firebase/firestore";
import { deleteObject, getDownloadURL, getStorage, ref, uploadString } from "firebase/storage";

// RTFeed's Firebase configuration
const firebaseConfig = {
    apiKey: process.env.REACT_APP_API_KEY,
    authDomain: process.env.REACT_APP_AUTH_DOMAIN,
    projectId: process.env.REACT_APP_PROJECT_ID,
    storageBucket: process.env.REACT_APP_STORAGE_BUCKET,
    messagingSenderId: process.env.REACT_APP_MESSAGIN_ID,
    appId: process.env.REACT_APP_APP_ID,
  };

export const app = initializeApp(firebaseConfig);

export const authService = getAuth(app);
export const authCreateUser = createUserWithEmailAndPassword;
export const authLogIn = signInWithEmailAndPassword;
export const onAuthChange = onAuthStateChanged;
export const authUpdateProfile = updateProfile;

// Social Login
export const authGoogleProvider = GoogleAuthProvider;
export const authGithubProvider = GithubAuthProvider;
export const authSignUpWithPopUp = signInWithPopup;

// Log Out
export const authLogOut = signOut;

// Database(Cloud Firestore)
export const dbService = getFirestore(app);
export const dbCollection = collection;
export const dbAddDoc = addDoc;
export const dbGetDocs = getDocs;
export const dbQuery = query;
export const dbOrderBy = orderBy;
export const dbDoc = doc;
export const dbDeleteDoc = deleteDoc;
export const dbUpdateDoc = updateDoc;
export const dbWhere = where;

// Realtime Listener
export const rtOnSnapshot = onSnapshot;

// Storage
export const storageService = getStorage(app);
export const storageRef = ref;
export const storageUploadString = uploadString;
export const storageGetDownloadURL = getDownloadURL;
// RefFromURL is gone.
export const storageDeleteObj = deleteObject;