// From SWPP Project. KJYOUNG copyright.
import { faX } from '@fortawesome/free-solid-svg-icons';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import styled from 'styled-components';
import { getContrastYIQ } from '../../styles/color';

interface IPropsTagBubble {
  color?: string;
  isPrime?: boolean;
}

interface IPropsTagBubbleX {
  testId?: string;
  onClick?: (e: React.MouseEvent) => void;
}

export const TagBubbleX = ({ testId, onClick }: IPropsTagBubbleX) => (
  <TagBubbleXstyle data-testid={testId} onClick={onClick}>
    <FontAwesomeIcon icon={faX} />
  </TagBubbleXstyle>
);

export const TagBubbleXstyle = styled.div`
  margin-left: 5px;
  font-size: 10px;
  color: red;
  width: fit-content;
  height: fit-content;
  display: block;
  cursor: pointer;
`;

export const TagBubbleWithFunc = styled.button<IPropsTagBubble>`
  height: 25px;
  border-radius: 30px;
  padding: 1px 10px;
  border: none;
  margin: 1px 2px;
  width: fit-content;
  display: flex;
  justify-content: space-between;
  align-items: center;
  ${({ color }) =>
    color &&
    `
      background: ${color};
      color: ${getContrastYIQ(color)}
    `}
`;

export const TagBubble = styled.button<IPropsTagBubble>`
  height: 20px;
  width: fit-content;
  border-radius: 25px;
  padding: 1px 12px;
  border: none;
  white-space: nowrap;
  margin: 1px 3px;
  ${({ color }) =>
    color &&
    `
      background: ${color};
      color: ${getContrastYIQ(color)};
    `}
  ${({ isPrime }) =>
    isPrime &&
    `
      border: 2px solid black;
    `}
`;

export const TagBubbleCompact = styled.button<IPropsTagBubble>`
  height: fit-content;
  width: fit-content;
  border-radius: 20px;
  padding: 4px 6px;
  border: none;
  white-space: nowrap;
  font-size: 11px;
  cursor: inherit;
  ${({ color }) =>
    color &&
    `
      background: ${color};
      color: ${getContrastYIQ(color)}
    `}
`;

export const TagBubbleCompactPointer = styled(TagBubbleCompact)`
  cursor: pointer;
`;

export const TagBubbleLarge = styled.button<IPropsTagBubble>`
  height: fit-content;
  width: fit-content;
  border-radius: 8px;
  padding: 4px 10px;
  margin: 1.5px 2px;
  border: none;
  white-space: nowrap;
  font-size: 11px;
  cursor: inherit;
  ${({ color }) =>
    color &&
    `
      background: ${color};
      color: ${getContrastYIQ(color)}
    `}
`;

export const TagBubbleHuge = styled.button<IPropsTagBubble>`
  height: fit-content;
  min-width: 60px;
  border-radius: 18px;
  padding: 8px 20px;
  margin: 0px 2px;
  border: none;
  white-space: nowrap;
  font-size: 20px;
  font-weight: 400;
  cursor: inherit;
  ${({ color }) =>
    color &&
    `
      background: ${color};
      color: ${getContrastYIQ(color, true)}
    `}
`;