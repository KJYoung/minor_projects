import { styled } from "styled-components";

export const DEFAULT_COLOR = "#000000";

export const getRandomHex = () => {
    return hslToHex(360 * Math.random(), 25 + 70 * Math.random(), 75 + 10 * Math.random());
};
  
export const hslToHex = (h: number, s: number, l: number) => {
    l /= 100;
    const a = (s * Math.min(l, 1 - l)) / 100;
    const f = (n: number) => {
        const k = (n + h / 30) % 12;
        const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
        return Math.round(255 * color)
        .toString(16)
        .padStart(2, '0'); // convert to Hex and prefix "0" if needed
    };
    return `#${f(0)}${f(8)}${f(4)}`;
};

export const getContrastYIQ = (hexcolor: string) => {
    hexcolor = hexcolor.replace('#', '');
    const r = parseInt(hexcolor.substring(0, 2), 16);
    const g = parseInt(hexcolor.substring(2, 4), 16);
    const b = parseInt(hexcolor.substring(4, 6), 16);
    const yiq = (r * 299 + g * 587 + b * 114) / 1000;
    return yiq >= 128 ? 'black' : 'white';
};
  
export const ColorCircle = styled.div<{ color: string, ishard?: string }>`
    position: relative;
    width: 20px;
    height: 20px;

    > div {
        cursor: pointer;
    }

    > div:first-child {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: ${props => ((props.ishard === 'true') ? '2px solid var(--ls-red)' : 'none')};
        background-color: ${props => (props.color)};;
        
        margin-right: 10px;

        display: flex;
        justify-content: center;
        align-items: center;
    }

    > div:last-child {
        position: absolute;
        top: 10px;
        left: 20px;
        width: 20px;
        height: 20px;
    }
`;

export const ColorCircleLarge = styled(ColorCircle)`
    width: 25px;
    height: 25px;

    margin-left: 8px;

    > div:first-child {
        width: 25px;
        height: 25px;
    }

    > div:last-child {
        position: absolute;
        top: 13px;
        left: 22px;
    }
`;