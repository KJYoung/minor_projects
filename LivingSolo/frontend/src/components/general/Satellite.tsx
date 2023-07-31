import { faCheck } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { styled } from "styled-components";
import { getContrastYIQ } from "../../styles/color";

interface SatelliteProps {
  colors: string[];
  doneNum: number;
  totalNum: number;
};

const top_left_array = [
  [], // 0
  [['38px', '13px']], // 1
  [ ['37px', '8px'], ['37px', '19px'],], // 2
  [
    ['35px', '2px'], ['38px', '13px'], ['35px', '24px']
  ], // 3 
  [
    ['31.5px', '-2.5px'], ['37px', '8px'], ['37px', '19px'], ['31.5px', '29.5px']
  ], // 4
  [
    ['28px', '-7px'], ['35px', '2px'], ['38px', '13px'], ['35px', '24px'], ['28px', '33px']
  ], // 5
  [
    ['22px', '-10px'], ['31.5px', '-2.5px'], ['37px', '8px'], ['37px', '19px'], ['31.5px', '29.5px'], ['22px', '36px']
  ], // 6
  [
    ['15px', '-12px'], ['26px', '-8px'], ['35px', '2px'], ['38px', '13px'], ['35px', '24px'], ['26px', '34px'], ['15px', '38px']
  ], // 7
];


export const SatelliteManualBottom = ({ colors: input_colors, totalNum, doneNum }: SatelliteProps) => { 
  const colors = (!input_colors || input_colors.length === 0) ? ['var(--ls-gray_lighter2)'] : (input_colors.length > 8 ? input_colors.slice(0, 8) : input_colors);
  return <SatelliteDiv>
      <SatelliteBig key={0} color={colors[0]}>
        {totalNum !== 0 && doneNum !== totalNum && <TodoNumIndicator>
            <TodoPendNumIndicator><span>{doneNum}</span></TodoPendNumIndicator>
            <TodoTotalNumIndicator><span>{totalNum}</span></TodoTotalNumIndicator>
        </TodoNumIndicator>}
        {totalNum !== 0 && doneNum === totalNum && <TodoCompleteCheckIndicator>
            <FontAwesomeIcon icon={faCheck} fontSize={'20'} color={getContrastYIQ(colors[0])}/>
          </TodoCompleteCheckIndicator>
        }
        {colors.map((color, idx) => {
          if(idx === 0){
            return <div key={idx}></div>
          }else{
            return <SatelliteSmall key={idx} color={color} top={(top_left_array[colors.length-1][idx-1][0])} left={(top_left_array[colors.length-1][idx-1][1])}></SatelliteSmall>
          }
        })}
      </SatelliteBig>
  </SatelliteDiv>
};

const TodoNumIndicator = styled.span`
  width: 38px;
  height: 38px;
  border-radius: 50%;
  background-color: transparent;

  display: flex;
  flex-direction: column;
  align-items: center;

  > span {
    width: 37px;
    height: 19px;
    background-color: transparent;
    color: black;

    display: flex;
    justify-content: center;
    align-items: center;
    font-size: 14px;

    padding-right: 2.5px;
  };
`;
const TodoPendNumIndicator = styled.span`
  border-radius: 38px 38px 0px 0px;
  padding-top: 4px;
`;

const TodoTotalNumIndicator = styled.span`
  border-radius: 0px 0px 38px 38px;
  padding-top: 1px;
  border-top: 1px solid var(--ls-gray);
`;

const TodoCompleteCheckIndicator = styled.div`
  display: flex;
  justify-content: center;
  align-items: center;
`;

const SatelliteBig = styled.div<{ color: string }>`
  width: 38px;
  height: 38px;
  border-radius: 50%;
  border: 1px solid var(--ls-gray);
  background-color: ${props => (props.color)};

  position: relative;

  display: flex;
  justify-content: center;
  align-items: center;
`;

const SatelliteSmall = styled.div<{ color: string, top: string, left: string }>`
  width: 10px;
  height: 10px;
  border-radius: 50%;
  border: 0.3px solid gray;
  background-color: ${props => (props.color)};

  position: absolute;
  top: ${props => (props.top)};
  left: ${props => (props.left)};
`;
const SatelliteDiv = styled.div`
  position: relative;
`;