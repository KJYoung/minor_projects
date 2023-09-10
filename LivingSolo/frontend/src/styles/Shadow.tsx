import { styled } from "styled-components";

export const ScrollShadow = styled.div`
  /* Scroll Shadow */
  background-image: linear-gradient(to top, white, white), linear-gradient(to top, white, white),
    linear-gradient(to top, rgba(0, 0, 0, 0.15), rgba(255, 255, 255, 0.3)),
    linear-gradient(to bottom, rgba(0, 0, 0, 0.15), rgba(255, 255, 255, 0.3));
  background-position: bottom center, top center, bottom center, top center;
  background-color: white;
  background-repeat: no-repeat;
  background-size: 100% 20px, 100% 20px, 100% 20px, 100% 20px;
  background-attachment: local, local, scroll, scroll;
`;